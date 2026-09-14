"""Tests for Worley noise: determinism, F1<=F2 ordering, neighbour vs brute, feature-point zeros, metrics."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import worley_noise as W  # noqa: E402


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
    # ---- 1. feature points are deterministic and vary across cells/seeds ----------------
    check("same cell+seed -> same points",
          W._cell_points(3, 5, 42, 2) == W._cell_points(3, 5, 42, 2))
    check("different cell -> different points",
          W._cell_points(3, 5, 42, 1) != W._cell_points(4, 5, 42, 1))
    check("different seed -> different points",
          W._cell_points(3, 5, 42, 1) != W._cell_points(3, 5, 43, 1))

    # ---- 2. feature points lie inside their cell ----------------------------------------
    pts = W._cell_points(7, -3, 1, 4)
    check("feature points inside their cell",
          all(7 <= x < 8 and -3 <= y < -2 for x, y in pts))

    # ---- 3. distances are sorted: F1 <= F2 <= F3 ----------------------------------------
    rnd = _lcg(7)
    ok = True
    for _ in range(2000):
        x, y = rnd() * 20, rnd() * 20
        d = W.worley(x, y, seed=1, n=3)
        if not (d[0] <= d[1] <= d[2] + 1e-12):
            ok = False
            break
    check("F1 <= F2 <= F3 always", ok)

    # ---- 4. the 3x3 neighbour computation misses nothing (== wide brute force) ----------
    rnd = _lcg(11)
    mism = 0
    for _ in range(1000):
        x, y = rnd() * 20, rnd() * 20
        a = W.worley(x, y, seed=1, n=2, radius=1)
        b = W.worley_brute(x, y, seed=1, n=2, radius=4)
        if abs(a[0] - b[0]) > 1e-12 or abs(a[1] - b[1]) > 1e-12:
            mism += 1
    check("radius=1 matches wide brute force", mism == 0, f"{mism}")
    # with several points per cell, radius=1 still suffices for F1
    rnd = _lcg(13)
    mism2 = 0
    for _ in range(500):
        x, y = rnd() * 20, rnd() * 20
        a = W.f1(x, y, seed=2, points_per_cell=3)
        b = W.worley_brute(x, y, seed=2, points_per_cell=3, n=1, radius=4)[0]
        if abs(a - b) > 1e-12:
            mism2 += 1
    check("multi-point-per-cell F1 matches brute", mism2 == 0, f"{mism2}")

    # ---- 5. F1 is exactly zero at a feature point ---------------------------------------
    fp = W._cell_points(2, 2, 1, 1)[0]
    check("F1 == 0 at a feature point", abs(W.f1(fp[0], fp[1], seed=1)) < 1e-12,
          f"{W.f1(fp[0], fp[1], seed=1)}")
    check("F1 > 0 away from feature points", W.f1(fp[0] + 0.3, fp[1] + 0.3, seed=1) > 0)

    # ---- 6. F2 - F1 is non-negative and small near boundaries ---------------------------
    rnd = _lcg(5)
    ok = all(W.f2_minus_f1(rnd() * 20, rnd() * 20, seed=1) >= -1e-12 for _ in range(500))
    check("F2 - F1 >= 0 everywhere", ok)

    # ---- 7. distance metrics: chebyshev <= euclidean <= manhattan (same pair) -----------
    check("metric ordering on a fixed pair",
          W._dist(0, 0, 3, 4, "chebyshev") <= W._dist(0, 0, 3, 4, "euclidean") <=
          W._dist(0, 0, 3, 4, "manhattan"))
    # euclidean of (3,4) is exactly 5
    check("euclidean (3,4) == 5", abs(W._dist(0, 0, 3, 4, "euclidean") - 5.0) < 1e-12)
    check("manhattan (3,4) == 7", abs(W._dist(0, 0, 3, 4, "manhattan") - 7.0) < 1e-12)
    check("chebyshev (3,4) == 4", abs(W._dist(0, 0, 3, 4, "chebyshev") - 4.0) < 1e-12)

    # ---- 8. all three metrics run and produce sorted output -----------------------------
    for metric in ("euclidean", "manhattan", "chebyshev"):
        d = W.worley(5.3, 7.1, seed=1, metric=metric, n=3)
        check(f"{metric} produces sorted F1<=F2<=F3", d[0] <= d[1] <= d[2])

    # ---- 9. field sampling produces the right shape and non-negative values -------------
    fld = W.field(16, 12, scale=4.0, seed=1, mode="f1")
    check("field has correct dimensions", len(fld) == 12 and len(fld[0]) == 16)
    check("field values are non-negative", all(v >= 0 for row in fld for v in row))
    fld2 = W.field(16, 12, scale=4.0, seed=1, mode="f2-f1")
    check("f2-f1 field non-negative", all(v >= -1e-12 for row in fld2 for v in row))

    # ---- 10. more points per cell lowers the average F1 ---------------------------------
    rnd = _lcg(9)
    samples = [(rnd() * 20, rnd() * 20) for _ in range(300)]
    avg1 = sum(W.f1(x, y, seed=3, points_per_cell=1) for x, y in samples) / len(samples)
    avg4 = sum(W.f1(x, y, seed=3, points_per_cell=4) for x, y in samples) / len(samples)
    check("denser feature points -> smaller mean F1", avg4 < avg1, f"1pc {avg1:.3f} 4pc {avg4:.3f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
