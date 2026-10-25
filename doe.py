"""Design of experiments on a simulated plasma etch step: 2^4 full factorial, ANOVA-style effect
tests, curvature check, optimisation and confirmation runs.

    python doe.py | tee doe_results.txt      # saves doe.png

The "process" is a hidden simulator with known true effects plus run-to-run noise, so the analysis
can be checked against the truth. Factors (coded -1 / +1):
    A  RF power          300 / 500 W
    B  chamber pressure  20 / 60 mTorr
    C  CF4 flow          50 / 100 sccm
    D  chuck temperature 20 / 60 C
Responses: etch rate (nm/min, target 420) and within-wafer non-uniformity (%, lower is better).
"""
import itertools
import sys
import numpy as np
from scipy import stats

FACTORS = ["A power", "B pressure", "C CF4 flow", "D temperature"]
LEVELS = [(300, 500, "W"), (20, 60, "mTorr"), (50, 100, "sccm"), (20, 60, "C")]
TERMS = ["A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD"]


def process(x, rng):
    """Hidden truth. x: coded settings (n, 4). Returns etch rate and non-uniformity with noise."""
    A, B, C, D = x.T
    rate = 380 + 55 * A + 22 * B + 14 * C + 0 * D + 12 * A * B - 9 * A ** 2 + rng.normal(0, 6, len(x))
    nonu = 4.0 + 0.9 * A - 0.7 * B + 0.0 * C + 0.3 * D + 0.6 * A * C + rng.normal(0, 0.25, len(x))
    return rate, nonu


TRUE = {"rate": {"A": 55, "B": 22, "C": 14, "AB": 12}, "nonu": {"A": 0.9, "B": -0.7, "D": 0.3, "AC": 0.6}}


def model_matrix(x, terms):
    cols = [np.ones(len(x))]
    for t in terms:
        cols.append(np.prod([x[:, "ABCD".index(ch)] for ch in t], axis=0))
    return np.column_stack(cols)


def fit(x, y, terms):
    M = model_matrix(x, terms)
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ beta
    dof = len(y) - M.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(M.T @ M)))
    tval = beta / se
    p = 2 * stats.t.sf(np.abs(tval), dof)
    return beta, se, p, s2, dof


def main(seed=7):
    rng = np.random.default_rng(seed)
    full = np.array(list(itertools.product([-1, 1], repeat=4)), float)
    design = np.vstack([full, full, np.zeros((4, 4))])        # 2 replicates + 4 centre points
    order = rng.permutation(len(design))                       # randomised run order
    design = design[order]
    rate, nonu = process(design, rng)
    fact = np.any(design != 0, axis=1)
    print(f"Design: 2^4 full factorial, 2 replicates, 4 centre points = {len(design)} runs in random order")
    out = {}
    for name, y, unit in (("rate", rate, "nm/min"), ("nonu", nonu, "%")):
        label = "etch rate" if name == "rate" else "non-uniformity"
        beta, se, p, s2, dof = fit(design[fact], y[fact], TERMS)
        print(f"\n[{label}]  residual sd {np.sqrt(s2):.2f} {unit} on {dof} degrees of freedom")
        print(f"  {'term':5s} {'effect':>9s} {'p':>9s}  {'significant':>11s}  true effect")
        sig = []
        for k, tm in enumerate(TERMS, start=1):
            eff = 2 * beta[k]                                  # effect = change from low to high
            true = 2 * TRUE[name].get(tm, 0)
            s = p[k] < 0.05
            sig += [tm] if s else []
            print(f"  {tm:5s} {eff:+9.2f} {p[k]:9.1e}  {'yes' if s else '':>11s}  {true:+.1f}")
        yc, yf = y[~fact], y[fact]
        tcurv = (yc.mean() - yf.mean()) / np.sqrt(s2 * (1 / len(yc) + 1 / len(yf)))
        pc = 2 * stats.t.sf(abs(tcurv), dof)
        print(f"  curvature (centre points vs factorial mean): {yc.mean() - yf.mean():+.2f} {unit}, p = {pc:.1e}"
              f"{'  -> curvature present, a 2-level model is only valid near the corners' if pc < 0.05 else ''}")
        keep = sorted(set(sig) | {ch for t in sig for ch in t}, key=TERMS.index)  # keep hierarchy
        out[name] = (keep, fit(design, y, keep)[0], sig, pc)
    # optimise: etch rate as close as possible to 420, then lowest non-uniformity
    grid = np.array(list(itertools.product(np.linspace(-1, 1, 9), repeat=4)))
    pr = model_matrix(grid, out["rate"][0]) @ out["rate"][1]
    pn = model_matrix(grid, out["nonu"][0]) @ out["nonu"][1]
    ok = np.abs(pr - 420) <= 5
    k = np.argmin(np.where(ok, pn, np.inf))
    best = grid[k]
    real = [lo + (v + 1) / 2 * (hi - lo) for v, (lo, hi, _) in zip(best, LEVELS)]
    print("\n[Optimum]  etch rate within 420 +/- 5 nm/min, lowest predicted non-uniformity")
    print("  " + ", ".join(f"{f.split()[1]} {r:.0f} {u}" for f, r, (_, _, u) in zip(FACTORS, real, LEVELS)))
    print(f"  predicted: etch rate {pr[k]:.1f} nm/min, non-uniformity {pn[k]:.2f} %")
    conf_r, conf_n = process(np.tile(best, (5, 1)), rng)
    print(f"  5 confirmation runs: etch rate {conf_r.mean():.1f} +/- {conf_r.std(ddof=1):.1f}, "
          f"non-uniformity {conf_n.mean():.2f} +/- {conf_n.std(ddof=1):.2f} %")
    base_n = process(np.zeros((5, 4)), rng)[1]
    print(f"  vs. the centre-point recipe: non-uniformity {base_n.mean():.2f} % -> {conf_n.mean():.2f} %")
    found = {n: sorted(out[n][2]) for n in out}
    truth = {n: sorted(TRUE[n]) for n in TRUE}
    print(f"\n  significant terms found: rate {found['rate']}, non-uniformity {found['nonu']}")
    print(f"  true terms:              rate {truth['rate']}, non-uniformity {truth['nonu']}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        for a, (name, y, lab) in zip(ax[:2], (("rate", rate, "etch rate"), ("nonu", nonu, "non-uniformity"))):
            beta, se, p, s2, dof = fit(design[fact], y[fact], TERMS)
            tv = np.abs(beta[1:] / se[1:]); o = np.argsort(tv)
            crit = stats.t.ppf(0.975, dof)
            a.barh(np.array(TERMS)[o], tv[o], color=np.where(tv[o] > crit, "#d62728", "0.6"))
            a.axvline(crit, color="k", ls="--", lw=1)
            a.set(title=f"Pareto of standardized effects: {lab}", xlabel="|t|  (dashed: p = 0.05)")
        a = ax[2]
        for bv, c in ((-1, "#1f77b4"), (1, "#d62728")):
            xs = np.array([-1, 1])
            m = model_matrix(np.column_stack([xs, [bv] * 2, [0] * 2, [0] * 2]), out["rate"][0]) @ out["rate"][1]
            a.plot([300, 500], m, "o-", color=c, label=f"pressure {20 if bv < 0 else 60} mTorr")
        a.set(title="A x B interaction on etch rate", xlabel="RF power (W)", ylabel="etch rate (nm/min)"); a.legend()
        fig.tight_layout(); fig.savefig("doe.png", dpi=105)
        print("saved doe.png")
    except ImportError:
        pass
    return found, truth


if __name__ == "__main__":
    found, truth = main()
    sys.exit(0 if found == truth else 1)
