"""Tests for Isolation Forest: outliers score higher, score bounds, c(n), reproducibility, dimensions."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import isolation_forest as IF  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main():
    rnd = _lcg(42)

    def gauss():
        return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())

    # dense cluster + planted outliers
    inliers = [[gauss() * 0.5, gauss() * 0.5] for _ in range(200)]
    outliers = [[6.0, 6.0], [-7.0, 5.0], [8.0, -6.0]]
    X = inliers + outliers
    forest = IF.IsolationForest(n_trees=120, sample_size=128, seed=1).fit(X)
    scores = forest.score_samples(X)

    # ---- 1. every planted outlier scores above the inlier mean --------------------------
    inlier_mean = sum(scores[:200]) / 200
    out_scores = scores[200:]
    check("outliers score above inlier mean", all(s > inlier_mean for s in out_scores),
          f"inlier {inlier_mean:.3f} out {[round(s,3) for s in out_scores]}")

    # ---- 2. top-ranked points are the outliers ------------------------------------------
    top3 = set(forest.rank(X)[:3])
    check("top-3 most anomalous are the planted outliers", top3 == {200, 201, 202}, f"{top3}")

    # ---- 3. scores lie in (0, 1) --------------------------------------------------------
    check("all scores in (0,1)", all(0.0 < s < 1.0 for s in scores))

    # ---- 4. a point far from all training data scores higher than a central one ---------
    central = forest.anomaly_score([0.0, 0.0])
    far = forest.anomaly_score([20.0, 20.0])
    check("far point scores higher than central", far > central, f"far {far:.3f} central {central:.3f}")

    # ---- 5. c(n) matches its closed form ------------------------------------------------
    # c(2) = 1
    check("c(2) == 1", abs(IF._c(2) - 1.0) < 1e-12)
    # c(n) = 2 H(n-1) - 2(n-1)/n
    def c_ref(n):
        H = sum(1.0 / k for k in range(1, n))
        return 2 * H - 2 * (n - 1) / n
    check("c(10) matches harmonic form", abs(IF._c(10) - c_ref(10)) < 1e-2, f"{IF._c(10)} vs {c_ref(10)}")
    check("c(1) == 0", IF._c(1) == 0.0)

    # ---- 6. reproducibility: same seed -> identical scores ------------------------------
    f1 = IF.IsolationForest(n_trees=50, sample_size=64, seed=7).fit(X)
    f2 = IF.IsolationForest(n_trees=50, sample_size=64, seed=7).fit(X)
    check("same seed -> identical scores",
          all(abs(f1.anomaly_score(X[i]) - f2.anomaly_score(X[i])) < 1e-12 for i in range(0, len(X), 20)))
    f3 = IF.IsolationForest(n_trees=50, sample_size=64, seed=99).fit(X)
    check("different seed -> different scores",
          any(abs(f1.anomaly_score(X[i]) - f3.anomaly_score(X[i])) > 1e-9 for i in range(len(X))))

    # ---- 7. path length is shorter for outliers than inliers ----------------------------
    pl_out = forest.path_length([8.0, -6.0])
    pl_in = forest.path_length([0.0, 0.0])
    check("outlier has shorter path length", pl_out < pl_in, f"out {pl_out:.2f} in {pl_in:.2f}")

    # ---- 8. 1-D data works --------------------------------------------------------------
    rnd2 = _lcg(3)
    x1 = [[rnd2() * 0.5] for _ in range(100)] + [[10.0]]
    f1d = IF.IsolationForest(n_trees=80, sample_size=64, seed=2).fit(x1)
    s1d = f1d.score_samples(x1)
    check("1-D: the far point is the top anomaly", f1d.rank(x1)[0] == 100, f"{f1d.rank(x1)[:3]}")

    # ---- 9. higher-dimensional data works -----------------------------------------------
    rnd3 = _lcg(11)
    x5 = [[rnd3() * 0.5 for _ in range(5)] for _ in range(150)]
    x5.append([9.0] * 5)
    f5 = IF.IsolationForest(n_trees=100, sample_size=100, seed=4).fit(x5)
    check("5-D: the far point is the top anomaly", f5.rank(x5)[0] == 150)

    # ---- 10. a uniform cloud has no strong anomalies (scores cluster near 0.5) ----------
    rnd4 = _lcg(5)
    uni = [[rnd4(), rnd4()] for _ in range(200)]
    fu = IF.IsolationForest(n_trees=100, sample_size=128, seed=6).fit(uni)
    su = fu.score_samples(uni)
    spread = max(su) - min(su)
    check("uniform cloud: scores are moderate (no extreme outlier)", max(su) < 0.75,
          f"max {max(su):.3f} spread {spread:.3f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
