"""SECOM yield root-cause study: data quality, yield trend, feature screening, SPC, models, report.

    python get_data.py                 # once: downloads the real dataset into data/
    python run_all.py | tee results.txt
    python run_all.py --data data_synthetic   # the planted-cause look-alike (python synth.py data_synthetic)
Writes results.png and report.md.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import analyze as A

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--boot", type=int, default=50)
args = ap.parse_args()
if not (Path(args.data) / "secom.data").exists():
    sys.exit(f"No {args.data}/secom.data. Run: python get_data.py   (or python synth.py data_synthetic)")

X, y, t = A.load(args.data)
print(f"[1  Data] {len(y)} wafers, {X.shape[1]} measurements, {t.min():%d %b %Y} to {t.max():%d %b %Y}")
print(f"  fails: {y.sum()} ({100 * y.mean():.1f} %), yield {100 * (1 - y.mean()):.1f} %")
cols, cs = A.clean(X)
print(f"  {100 * cs['cells_missing']:.1f} % of all values missing; dropped {cs['too_missing']} features over 50 % missing "
      f"and {cs['constant']} that never change; {cs['kept']} features kept")

weekly = pd.DataFrame({"t": t, "y": y}).set_index("t").resample("W")["y"].agg(["mean", "count"])
weekly = weekly[weekly["count"] >= 20]
worst = weekly["mean"].idxmax()
print(f"  weekly fail rate ranges {100 * weekly['mean'].min():.1f}-{100 * weekly['mean'].max():.1f} % "
      f"(worst week starting {worst - pd.Timedelta(days=6):%d %b}, {int(weekly.loc[worst, 'count'])} wafers)")

print("\n[2  Screening every feature: fails vs. passes]")
S = A.screen(X, y, cols)
sig = S[S["q"] < 0.10]
print(f"  Mann-Whitney U test per feature, Benjamini-Hochberg false-discovery rate across {len(S)} tests")
print(f"  raw p < 0.05: {int((S['p'] < 0.05).sum())} features (about {0.05 * len(S):.0f} expected by chance alone); "
      f"q < 0.10 after correction: {len(sig)}")
print(f"  {'feature':8s} {'p':>9s} {'q':>7s} {'AUC':>6s} {'fails run':>10s} {'missing':>8s}")
for _, r in S.head(10).iterrows():
    print(f"  {r.feature:8s} {r.p:9.2e} {r.q:7.3f} {r.auc:6.3f} {r.fails_run:>10s} {100 * r.missing:7.1f}%")

print("\n[3  SPC: I-MR charts, limits from the first 30 % of production, Western Electric rules 1-3]")
top = list(S.head(10)["feature"])
P = A.spc_table(X, y, top).set_index("feature")
print(f"  {'feature':8s} {'out of control':>15s} {'fail rate when OOC':>19s} {'fail rate in control':>21s}")
for c in top:
    if c in P.index:
        r = P.loc[c]
        fo = "-" if pd.isna(r.fail_rate_ooc) else f"{100 * r.fail_rate_ooc:.1f} %"
        print(f"  {c:8s} {100 * r.ooc_rate:13.1f} % {fo:>19s} {100 * r.fail_rate_in:19.1f} %")

drifted = [c for c in top if c in P.index and P.loc[c, "ooc_rate"] > 0.25]
if drifted:
    print(f"  {len(drifted)} of the top {len(top)} are out of control on more than 25 % of later wafers: the process moved away "
          "from its baseline after the first 30 % of production")
early, late = y.iloc[: int(0.3 * len(y))].mean(), y.iloc[int(0.3 * len(y)):].mean()
print(f"  fail rate in the baseline period {100 * early:.1f} % vs. after it {100 * late:.1f} %")

print("\n[4  Predicting fails: train on the first 70 % of production, test on the last 30 %]")
cut = int(0.7 * len(y))
Xtr, Xte, ytr, yte = X[cols].iloc[:cut], X[cols].iloc[cut:], y.iloc[:cut], y.iloc[cut:]
print(f"  train {len(ytr)} wafers ({ytr.sum()} fails), test {len(yte)} wafers ({yte.sum()} fails); "
      f"a random guess scores PR-AUC {yte.mean():.3f}")
res = {}
for name, m in (("L1 logistic regression", A.lasso_logit()), ("random forest", A.forest())):
    res[name] = A.evaluate(m, Xtr, ytr, Xte, yte); res[name]["model"] = m
    r = res[name]
    print(f"  {name:24s} ROC-AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  "
          f"fails caught when flagging the top 10 % of wafers: {100 * r['recall_at_10']:.0f} %")
best_name = max(res, key=lambda k: res[k]["pr_auc"])
imp = A.perm_importance(res[best_name]["model"], Xte, yte)
print(f"  permutation importance ({best_name}, drop in PR-AUC when a feature is shuffled): "
      + ", ".join(f"{c} {v:+.3f}" for c, v in imp.head(6).items()))
sel = A.bootstrap_selection(Xtr, ytr, cols, n_boot=args.boot)
print(f"  L1 selection stability over {args.boot} bootstrap resamples: "
      + ", ".join(f"{c} {100 * v:.0f} %" for c, v in sel.head(6).items()))

best = res[best_name]
skilful = best["pr_auc"] >= 1.5 * yte.mean() and best["roc_auc"] >= 0.6
if not skilful:
    print(f"  the models barely beat chance on later wafers (best ROC-AUC {best['roc_auc']:.2f}, PR-AUC {best['pr_auc']:.3f} "
          f"vs {yte.mean():.3f}): pass/fail patterns from early production do not carry forward, so the model-based "
          "rankings are not used as evidence below")

if skilful:
    print("\n[5  Suspects: flagged by at least two of: screening q < 0.05, L1 picked in >= 50 % of bootstraps, permutation importance >= 10 % of the top]")
else:
    print("\n[5  Suspects: screening only (q < 0.05, top 10 by p), because the models had no forward skill]")
votes = {}
for c in S[S["q"] < 0.05]["feature"].head(20 if skilful else 10):   # significant after FDR correction
    votes.setdefault(c, []).append("screening")
if skilful:
    for c in sel[sel >= 0.5].index:                               # picked in at least half the bootstraps
        votes.setdefault(c, []).append("stable L1")
    for c, v in imp.items():                                      # at least 10 % of the top importance
        if v > 0 and v >= 0.1 * imp.max():
            votes.setdefault(c, []).append("permutation")
need = 2 if skilful else 1
suspects = sorted([c for c, v in votes.items() if len(v) >= need], key=lambda c: (-len(votes[c]), S.set_index("feature").loc[c, "p"] if c in set(S["feature"]) else 1))
St = S.set_index("feature")
rho = X[suspects].corr(method="spearman") if suspects else None
proxy = {}
for k, c in enumerate(suspects):
    r = St.loc[c]
    lead = max(suspects[:k], key=lambda d: abs(rho.loc[c, d]), default=None)
    note = ""
    if lead is not None and abs(rho.loc[c, lead]) >= 0.4:
        proxy[c] = lead
        note = f"; moves with {lead} (Spearman {rho.loc[c, lead]:+.2f}), likely the same underlying cause"
    print(f"  {c}: {', '.join(votes[c])}; fails run {r.fails_run} (median {r.median_fail:.4g} vs {r.median_pass:.4g}), AUC {r.auc:.3f}{note}")
primary = [c for c in suspects if c not in proxy]
print(f"  independent suspects after grouping correlated sensors: {', '.join(primary) if primary else 'none'}")
if not suspects:
    print("  none: no feature met the rule")

FAILED = False
planted = Path(args.data) / "planted.json"
if planted.exists():
    pl = json.loads(planted.read_text())
    found = [c for c in pl if c in primary]
    print(f"\n  synthetic check: planted causes {pl}, recovered {found}")
    if len(found) < len(pl):
        print("  FAIL: not every planted cause was recovered"); FAILED = True

# ---------------------------------------------------------------- report
lines = [f"# SECOM yield root-cause report", "",
         f"**Data:** {len(y)} wafers, {t.min():%d %b} to {t.max():%d %b %Y}; {y.sum()} fails ({100 * y.mean():.1f} %). "
         f"{cs['kept']} of {cs['features']} measurements usable after removing {cs['too_missing']} mostly-missing and {cs['constant']} constant ones.", "",
         "## In plain words", ""]
if primary:
    lines += [f"Out of {len(S)} measurements, **{len(primary)}** independent ones stand out "
              + ("by more than one method: " if skilful else "in the pass/fail screening: ")
              + ", ".join(f"**{c}** (fails run {St.loc[c, 'fails_run']})" for c in primary) + ". "
              + (f"{len(proxy)} more move together with these and are probably the same causes seen by another sensor. " if proxy else "")
              + 
              "These are the parameters to check first in the tool logs and maintenance records for the affected period. "
              "This is statistical evidence, not proof of cause; a controlled experiment (DOE) on the suspected parameters would confirm it.", ""]
lines += [f"A model trained on the first 70 % of production and tested on the last 30 % ranks wafers so that flagging the top 10 % "
          f"catches **{100 * res[best_name]['recall_at_10']:.0f} %** of fails ({best_name}, PR-AUC {res[best_name]['pr_auc']:.3f} vs {yte.mean():.3f} for chance)."
          + ("" if skilful else " That is close to chance: the patterns behind early failures did not carry forward"
             + (", and the SPC charts show the process moving after the baseline period. Re-baselining the control limits and "
                "monitoring the suspects going forward is the practical next step." if drifted else ".")), "",
          "## Top features by pass/fail screening", "", "| Feature | p | q (FDR) | AUC | Fails run |", "|---|---|---|---|---|"]
lines += [f"| {r.feature} | {r.p:.2e} | {r.q:.3f} | {r.auc:.3f} | {r.fails_run} |" for _, r in S.head(10).iterrows()]
Path("report.md").write_text("\n".join(lines) + "\n")
print("\nwrote report.md")

# ---------------------------------------------------------------- plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a = ax[0, 0]
    a.bar(weekly.index, 100 * weekly["mean"], width=5, color="#d62728")
    a.axhline(100 * y.mean(), color="k", ls="--", lw=1, label="overall")
    a.set(title="Weekly fail rate", ylabel="% of wafers failing"); a.legend(); a.tick_params(axis="x", rotation=30)
    a = ax[0, 1]
    a.hist(100 * X.isna().mean(), bins=40, color="0.5")
    a.axvline(50, color="k", ls="--", lw=1)
    a.set(title="Missing data per measurement", xlabel="% missing", ylabel="measurements", yscale="log")
    a = ax[0, 2]
    s = S.head(12)[::-1]
    a.barh(s["feature"], s["auc"] - 0.5, color=np.where(s["auc"] > 0.5, "#d62728", "#1f77b4"))
    a.axvline(0, color="k", lw=0.8)
    a.set(title="Top features: AUC - 0.5 (red: fails run high)", xlabel="separation of fails from passes")
    c0 = suspects[0] if suspects else S.iloc[0]["feature"]
    centre, sigma, ooc = A.imr_chart(X[c0])
    a = ax[1, 0]
    idx = np.arange(len(y))
    a.plot(idx, X[c0], ".", ms=2, color="0.6", label="wafer")
    a.plot(idx[y == 1], X[c0][y == 1], "x", color="#d62728", ms=4, label="fail")
    for k, ls in ((0, "-"), (3, "--"), (-3, "--")):
        a.axhline(centre + k * sigma, color="k", ls=ls, lw=1)
    a.axvspan(0, 0.3 * len(y), color="#1f77b4", alpha=0.08, label="baseline for limits")
    a.set(title=f"I chart for {c0} (3-sigma limits)", xlabel="wafer, in time order"); a.legend(fontsize=8)
    a = ax[1, 1]
    for name, col in zip(res, ("#1f77b4", "#ff7f0e")):
        pr, rc, _ = precision_recall_curve(yte, res[name]["scores"])
        a.plot(rc, pr, color=col, label=f"{name} (AP {res[name]['pr_auc']:.2f})")
    a.axhline(yte.mean(), color="k", ls=":", label="chance")
    a.set(title="Fail prediction on the last 30 % of production", xlabel="recall (fails caught)", ylabel="precision"); a.legend(fontsize=8)
    a = ax[1, 2]
    h = sel.head(12)[::-1]
    a.barh(h.index, 100 * h.values, color="#2ca02c")
    a.set(title=f"L1 selection frequency ({args.boot} bootstraps)", xlabel="% of resamples selecting the feature")
    fig.tight_layout(); fig.savefig("results.png", dpi=105)
    print("saved results.png")
except ImportError:
    print("matplotlib not installed; skipped the plot")
sys.exit(1 if FAILED else 0)
