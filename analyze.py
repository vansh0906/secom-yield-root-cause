"""Building blocks for the SECOM yield analysis: loading, cleaning, screening, SPC and models."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load(data_dir="data"):
    d = Path(data_dir)
    X = pd.read_csv(d / "secom.data", sep=r"\s+", header=None, na_values=["NaN", "nan"])
    X.columns = [f"F{j:03d}" for j in range(X.shape[1])]
    lab = pd.read_csv(d / "secom_labels.data", sep=r"\s+", header=None, names=["y", "t"], quotechar='"')
    t = pd.to_datetime(lab["t"], format="%d/%m/%Y %H:%M:%S")
    y = (lab["y"] == 1).astype(int)                       # 1 = fail
    order = np.argsort(t.values, kind="stable")
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), t.iloc[order].reset_index(drop=True)


def clean(X, max_missing=0.5):
    """Drop features that are mostly missing or never change. Returns kept columns and a summary."""
    miss = X.isna().mean()
    nun = X.nunique(dropna=True)
    too_missing = miss[miss > max_missing].index
    constant = nun[nun <= 1].index.difference(too_missing)
    keep = [c for c in X.columns if c not in set(too_missing) | set(constant)]
    return keep, {"features": X.shape[1], "too_missing": len(too_missing), "constant": len(constant),
                  "kept": len(keep), "cells_missing": float(X.isna().values.mean())}


def bh_fdr(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1)
    return out


def screen(X, y, cols):
    """Pass vs. fail comparison for every feature: Mann-Whitney U, AUC and FDR-corrected q-value."""
    rows = []
    for c in cols:
        x = X[c]; ok = x.notna()
        a, b = x[ok & (y == 1)], x[ok & (y == 0)]
        if len(a) < 5 or len(b) < 5:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))                        # P(fail value > pass value)
        rows.append((c, p, auc, a.median(), b.median(), float(x.isna().mean())))
    df = pd.DataFrame(rows, columns=["feature", "p", "auc", "median_fail", "median_pass", "missing"])
    df["q"] = bh_fdr(df["p"].values)
    df["effect"] = (df["auc"] - 0.5).abs()
    df["fails_run"] = np.where(df["auc"] > 0.5, "high", "low")
    return df.sort_values("p").reset_index(drop=True)


def imr_chart(x, baseline_frac=0.3):
    """Individuals (I-MR) control chart with Western Electric rules 1, 2 and 3.

    Limits come from the first `baseline_frac` of production (in time order), with sigma
    estimated from the average moving range (MR-bar / 1.128), the textbook I-MR method.
    Returns centre, sigma and a boolean out-of-control flag per point.
    """
    x = pd.Series(x).astype(float)
    base = x.iloc[: int(len(x) * baseline_frac)].dropna()
    centre = base.mean()
    sigma = base.diff().abs().mean() / 1.128
    v = x.values
    ooc = np.zeros(len(v), bool)
    ooc |= np.abs(v - centre) > 3 * sigma                                       # rule 1
    side = np.sign(v - centre)
    run = 0
    for k in range(len(v)):                                                     # rule 2: 9 on one side
        run = run + 1 if k and side[k] == side[k - 1] and side[k] != 0 and not np.isnan(v[k]) else 1
        if run >= 9:
            ooc[k] = True
    inc = 0
    for k in range(1, len(v)):                                                  # rule 3: 6 rising/falling
        d = np.sign(v[k] - v[k - 1])
        inc = inc + 1 if d != 0 and k > 1 and d == np.sign(v[k - 1] - v[k - 2]) else 1
        if inc >= 6:
            ooc[k] = True
    ooc &= ~np.isnan(v)
    return centre, sigma, ooc


def spc_table(X, y, cols, baseline_frac=0.3):
    rows = []
    later = np.arange(len(y)) >= int(len(y) * baseline_frac)
    for c in cols:
        centre, sigma, ooc = imr_chart(X[c], baseline_frac)
        if not np.isfinite(sigma) or sigma == 0:
            continue
        m = later & X[c].notna().values
        o, i = m & ooc, m & ~ooc
        fr_o = y[o].mean() if o.sum() >= 10 else np.nan
        fr_i = y[i].mean() if i.sum() else np.nan
        rows.append((c, centre, sigma, o.sum() / max(m.sum(), 1), fr_o, fr_i))
    return pd.DataFrame(rows, columns=["feature", "centre", "sigma", "ooc_rate", "fail_rate_ooc", "fail_rate_in"])


# ---------------------------------------------------------------- predictive models
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance


def lasso_logit(C=0.05, seed=0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                            class_weight="balanced", random_state=seed))


def forest(seed=0):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed))


def recall_at_flag_rate(y, score, rate=0.10):
    """Share of fails caught if the top `rate` of wafers by score are flagged for inspection."""
    k = max(1, int(round(rate * len(y))))
    top = np.argsort(-score)[:k]
    return float(np.asarray(y)[top].sum() / max(np.asarray(y).sum(), 1))


def evaluate(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    s = model.predict_proba(Xte)[:, 1]
    return {"roc_auc": roc_auc_score(yte, s), "pr_auc": average_precision_score(yte, s),
            "recall_at_10": recall_at_flag_rate(yte, s, 0.10), "scores": s}


def bootstrap_selection(Xtr, ytr, cols, n_boot=50, C=0.05, seed=0):
    """How often each feature gets a non-zero L1 coefficient across bootstrap resamples."""
    rng = np.random.default_rng(seed)
    counts = pd.Series(0, index=cols, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(ytr), len(ytr))
        m = lasso_logit(C, seed=b).fit(Xtr.iloc[idx], ytr.iloc[idx])
        coef = m[-1].coef_[0]
        counts[np.array(cols)[coef != 0]] += 1
    return (counts / n_boot).sort_values(ascending=False)


def perm_importance(model, Xte, yte, seed=0, top=15):
    r = permutation_importance(model, Xte, yte, scoring="average_precision", n_repeats=10, random_state=seed, n_jobs=-1)
    return pd.Series(r.importances_mean, index=Xte.columns).sort_values(ascending=False).head(top)
