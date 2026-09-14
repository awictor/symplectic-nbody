"""Tests for Local Outlier Factor: inliers ~1, global + LOCAL outliers flagged, reachability properties."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lof as L  # noqa: E402


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

    # ---- 1. single cluster + a far global outlier ---------------------------------------
    X = [[gauss() * 0.5, gauss() * 0.5] for _ in range(120)]
    X.append([6.0, 6.0])
    lof = L.LOF(X, k=20)
    sc = lof.scores()
    inlier_mean = sum(sc[:120]) / 120
    check("inlier LOF near 1", 0.8 < inlier_mean < 1.3, f"{inlier_mean:.3f}")
    check("global outlier LOF >> 1", sc[120] > 2.0, f"{sc[120]:.3f}")
    check("outlier is top-ranked", lof.rank()[0] == 120)

    # ---- 2. uniform cloud: no strong outliers, LOF clusters near 1 ----------------------
    rnd2 = _lcg(7)
    uni = [[rnd2(), rnd2()] for _ in range(200)]
    lu = L.LOF(uni, k=20)
    su = lu.scores()
    # most interior points have LOF ~ 1; a few boundary points slightly higher
    frac_near1 = sum(1 for s in su if s < 1.5) / len(su)
    check("uniform cloud: most LOF < 1.5", frac_near1 > 0.85, f"{frac_near1:.2f}")

    # ---- 3. THE local-outlier property: dense + sparse clusters -------------------------
    # a point that is a normal DISTANCE for the sparse cluster but abnormal next to the dense one
    rnd3 = _lcg(11)

    def g3():
        return math.sqrt(-2 * math.log(rnd3() + 1e-12)) * math.cos(2 * math.pi * rnd3())

    dense = [[g3() * 0.3, g3() * 0.3] for _ in range(80)]
    sparse = [[g3() * 2 + 12, g3() * 2 + 12] for _ in range(40)]
    local_out = [2.2, 2.2]                                # just outside the dense cluster
    pts = dense + sparse + [local_out]
    lof3 = L.LOF(pts, k=15)
    s3 = lof3.scores()
    dense_mean = sum(s3[:80]) / 80
    check("dense-cluster points have LOF ~ 1", dense_mean < 1.4, f"{dense_mean:.3f}")
    check("local outlier is flagged (LOF > 1.5)", s3[-1] > 1.5, f"{s3[-1]:.3f}")
    # the key discriminator: local_out's raw distance to the dense cluster is comparable to
    # typical spacing INSIDE the sparse cluster, yet LOF still flags it
    sparse_typical_spacing = L.LOF(pts, k=15).k_distance(80)   # a sparse point's k-distance
    dist_to_dense = min(math.dist(local_out, dense[i]) for i in range(80))
    check("local outlier distance is 'normal' by sparse-cluster standards",
          dist_to_dense < sparse_typical_spacing * 1.5,
          f"d {dist_to_dense:.2f} vs sparse kdist {sparse_typical_spacing:.2f}")

    # ---- 4. reachability distance >= raw distance and >= k-distance floor ---------------
    i, j = 0, 5
    rd = lof.reachability_distance(i, j)
    raw = math.dist(X[i], X[j])
    check("reachability distance >= raw distance", rd >= raw - 1e-12, f"{rd} vs {raw}")
    check("reachability distance >= k-distance floor", rd >= lof.k_distance(j) - 1e-12)

    # ---- 5. local reachability density is positive for a normal point -------------------
    check("lrd positive for an inlier", lof.local_reachability_density(0) > 0)

    # ---- 6. score list length and convenience function ----------------------------------
    conv = L.lof_scores(X, k=20)
    check("lof_scores convenience matches class",
          all(abs(conv[i] - sc[i]) < 1e-12 for i in range(0, len(X), 25)))
    check("one score per point", len(sc) == len(X))

    # ---- 7. a tight near-duplicate group plus an isolated point -------------------------
    # (exact duplicates give zero k-distance -> infinite density, a documented LOF edge case;
    # a tight jittered group keeps densities finite while staying the analogous scenario)
    grp = [[rnd() * 1e-3, rnd() * 1e-3] for _ in range(8)]
    grp.append([5.0, 5.0])                               # isolated point
    ld = L.LOF(grp, k=3)
    sd = ld.scores()
    check("tight-group + isolated: all scores finite", all(math.isfinite(s) for s in sd), f"{sd}")
    check("isolated point ranks most anomalous", ld.rank()[0] == 8, f"{ld.rank()[:3]}")

    # ---- 8. higher dimensions ----------------------------------------------------------
    rnd4 = _lcg(13)
    x5 = [[rnd4() * 0.5 for _ in range(5)] for _ in range(120)]
    x5.append([8.0] * 5)
    l5 = L.LOF(x5, k=20)
    check("5-D: far point is top anomaly", l5.rank()[0] == 120, f"{l5.rank()[:3]}")

    # ---- 9. two well-separated equal-density clusters: no false anomalies ---------------
    rnd5 = _lcg(17)

    def g5():
        return math.sqrt(-2 * math.log(rnd5() + 1e-12)) * math.cos(2 * math.pi * rnd5())

    c1 = [[g5() * 0.5 - 5, g5() * 0.5] for _ in range(80)]
    c2 = [[g5() * 0.5 + 5, g5() * 0.5] for _ in range(80)]
    lc = L.LOF(c1 + c2, k=20)
    sc2 = lc.scores()
    check("two equal-density clusters: few false positives",
          sum(1 for s in sc2 if s > 2.0) < 8, f"{sum(1 for s in sc2 if s > 2.0)} flagged")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
