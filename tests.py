"""Quick self-checks. Run: python tests.py"""
import numpy as np
import pandas as pd
import analyze as A

passed = 0


def test(name, cond):
    global passed
    assert cond, name
    passed += 1
    print(f"ok  {name}")


# Benjamini-Hochberg against a hand-worked example
p = np.array([0.01, 0.04, 0.03, 0.005])
test("BH q-values match the hand calculation", np.allclose(A.bh_fdr(p), [0.02, 0.04, 0.04, 0.02]))

# screening direction and strength
rng = np.random.default_rng(0)
n = 400
y = pd.Series((rng.random(n) < 0.1).astype(int))
X = pd.DataFrame({"hi": rng.normal(size=n) + 1.5 * y, "lo": rng.normal(size=n) - 1.5 * y, "noise": rng.normal(size=n)})
S = A.screen(X, y, list(X.columns)).set_index("feature")
test("screening finds a feature that runs high in fails", S.loc["hi", "auc"] > 0.8 and S.loc["hi", "q"] < 1e-6)
test("screening finds a feature that runs low in fails", S.loc["lo", "auc"] < 0.2 and S.loc["lo", "fails_run"] == "low")
test("pure noise is not significant", S.loc["noise", "q"] > 0.05)

# I-MR chart rules
x = np.r_[rng.normal(0, 1, 300), rng.normal(0, 1, 50) + 2.0]
c, s, ooc = A.imr_chart(x, 0.5)
test("I-MR limits estimate sigma from the moving range", abs(s - 1) < 0.15 and abs(c) < 0.2)
test("a 2-sigma shift is flagged by the run rule", ooc[320:].mean() > 0.3 and ooc[:300].mean() < 0.05)
spike = np.r_[np.zeros(50) + rng.normal(0, 0.1, 50), 5.0, rng.normal(0, 0.1, 10)]
test("a single point beyond 3 sigma is flagged", A.imr_chart(spike, 0.5)[2][50])

# cleaning
Xc = pd.DataFrame({"ok": rng.normal(size=10), "const": np.ones(10), "gaps": [np.nan] * 8 + [1.0, 2.0]})
keep, info = A.clean(Xc)
test("cleaning drops constant and mostly-missing features", keep == ["ok"] and info["constant"] == 1 and info["too_missing"] == 1)

# recall at flag rate
yy = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]); sc = np.array([.9, .8, .1, .7, .1, .1, .1, .1, .1, .1])
test("recall at a 20 % flag rate", A.recall_at_flag_rate(yy, sc, 0.2) == 0.5)

# DOE recovers the hidden effects
import doe
found, truth = doe.main()
test("DOE finds exactly the true significant effects", found == truth)
print(f"\n{passed} tests passed")
