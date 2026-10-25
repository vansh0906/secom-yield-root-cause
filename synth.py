"""A SECOM look-alike with known root causes, used by tests and CI.

Same shape and file format as the real data: 1,567 wafers x 590 measurements, NaN gaps,
constant columns, wildly different scales, about 6-7 % fails and timestamps over three months.
Three sensors are planted as causes, so the pipeline can be checked for recovering them:
  F059  fails when it runs high
  F103  drifts upward in the second half of the period, and fails follow the drift
  F433  fails when it runs low
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

PLANTED = ["F059", "F103", "F433"]


def make(out_dir, seed=0, n=1567, p=590):
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2008-07-19 11:55") + pd.to_timedelta(np.sort(rng.uniform(0, 90 * 24 * 3600, n)), unit="s")
    latent = rng.normal(size=(n, 12))
    load = rng.normal(size=(12, p)) * (rng.random((12, p)) < 0.15)
    X = latent @ load + rng.normal(size=(n, p))
    frac = np.arange(n) / n
    X[:, 103] += 2.5 * np.clip(frac - 0.5, 0, None) / 0.5          # slow drift after mid-period
    z = 1.3 * X[:, 59] / X[:, 59].std() + 1.6 * (X[:, 103] - X[:, 103][: n // 2].mean()) - 1.1 * X[:, 433] / X[:, 433].std()
    logit = -6.0 + z
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), 1, -1)
    scale = 10 ** rng.uniform(-3, 3, p); offset = rng.uniform(-5, 5, p) * scale
    X = X * scale + offset
    const = rng.choice([j for j in range(p) if j not in (59, 103, 433)], 116, replace=False)
    X[:, const] = np.round(offset[const], 3)
    sparse = rng.choice([j for j in range(p) if j not in (59, 103, 433)], 28, replace=False)
    X[rng.random((n, p)) < 0.03] = np.nan
    for j in sparse:
        X[rng.random(n) < rng.uniform(0.5, 0.92), j] = np.nan
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "secom.data", X, fmt="%.6g")
    with open(out / "secom_labels.data", "w") as f:
        for yi, ti in zip(y, t):
            f.write(f'{yi} "{ti.strftime("%d/%m/%Y %H:%M:%S")}"\n')
    (out / "planted.json").write_text(json.dumps(PLANTED))
    return int((y == 1).sum())


if __name__ == "__main__":
    import sys
    print("fails:", make(sys.argv[1] if len(sys.argv) > 1 else "data_synthetic"))
