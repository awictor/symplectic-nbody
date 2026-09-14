"""Tests for soft-DTW: gamma->0 = hard DTW, symmetry, softmin bounds, alignment matrix, smoothness."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import soft_dtw as S  # noqa: E402


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


def main():
    x = [0, 1, 3, 2, 5, 4]
    y = [1, 0, 2, 4, 3, 5, 4]

    # ---- 1. gamma -> 0 converges to hard DTW -------------------------------------------
    hd = S.hard_dtw(x, y)
    diffs = [abs(S.soft_dtw(x, y, g) - hd) for g in (0.1, 0.01, 0.001)]
    check("soft-DTW -> hard DTW as gamma -> 0", diffs[0] > diffs[1] > diffs[2] and diffs[2] < 1e-2,
          f"diffs {diffs}")
    check("soft-DTW at gamma=0.001 ~ hard DTW", abs(S.soft_dtw(x, y, 0.001) - hd) < 1e-2)

    # ---- 2. symmetry --------------------------------------------------------------------
    check("soft-DTW is symmetric", abs(S.soft_dtw(x, y, 1.0) - S.soft_dtw(y, x, 1.0)) < 1e-9)
    check("hard DTW is symmetric", abs(S.hard_dtw(x, y) - S.hard_dtw(y, x)) < 1e-9)

    # ---- 3. self-distance is minimal ----------------------------------------------------
    check("soft-DTW(x,x) <= soft-DTW(x,y)", S.soft_dtw(x, x, 0.01) <= S.soft_dtw(x, y, 0.01) + 1e-9)
    check("hard DTW(x,x) == 0", abs(S.hard_dtw(x, x)) < 1e-9)

    # ---- 4. softmin bounds: min - gamma*log(3) <= softmin <= min ------------------------
    for (a, b, c, g) in [(1, 2, 3, 0.5), (0.1, 0.2, 5.0, 1.0), (2, 2, 2, 0.3)]:
        sm = S._softmin(a, b, c, g)
        lo = min(a, b, c) - g * math.log(3)
        hi = min(a, b, c)
        check(f"softmin({a},{b},{c},g={g}) in [min-g*log3, min]", lo - 1e-9 <= sm <= hi + 1e-9,
              f"{sm} not in [{lo:.4f},{hi:.4f}]")

    # ---- 5. softmin -> min as gamma -> 0 ------------------------------------------------
    check("softmin -> min as gamma -> 0", abs(S._softmin(1, 5, 3, 1e-4) - 1) < 1e-3)

    # ---- 6. alignment matrix: non-negative, endpoints occupied --------------------------
    E = S.alignment_matrix(x, y, 1.0)
    check("alignment matrix has right shape", len(E) == len(x) and len(E[0]) == len(y))
    check("alignment matrix non-negative", all(v >= -1e-12 for row in E for v in row))
    check("alignment starts at (0,0)", E[0][0] > 0.5, f"{E[0][0]}")
    check("alignment ends at (n,m)", E[-1][-1] > 0.5, f"{E[-1][-1]}")

    # ---- 7. alignment concentrates on the optimal path as gamma -> 0 --------------------
    # at small gamma the alignment matrix mass should lie on the hard-DTW path (each row/col
    # near a 0/1 pattern); measure by how close the max entry per row is to 1
    E_small = S.alignment_matrix(x, y, 0.001)
    # every row should have an entry near an integer occupancy (0 or the path count)
    concentrated = sum(1 for row in E_small if max(row) > 0.9)
    check("alignment concentrates as gamma -> 0", concentrated >= len(x) - 1,
          f"{concentrated}/{len(x)} rows concentrated")

    # ---- 8. smoothness: a tiny perturbation changes soft-DTW by a tiny amount -----------
    base = S.soft_dtw(x, y, 1.0)
    xp = [xi + 1e-6 for xi in x]
    pert = S.soft_dtw(xp, y, 1.0)
    check("soft-DTW is smooth (small perturbation -> small change)", abs(pert - base) < 1e-2,
          f"delta {abs(pert - base)}")

    # ---- 9. shift invariance under warping: a time-shifted copy aligns cheaply ----------
    a = [math.sin(2 * math.pi * k / 12) for k in range(24)]
    b = [math.sin(2 * math.pi * (k - 2) / 12) for k in range(24)]  # shifted by 2 samples
    # DTW should align them far better than the naive Euclidean distance
    dtw_cost = S.hard_dtw(a, b)
    eucl = sum((a[k] - b[k]) ** 2 for k in range(24))
    check("DTW aligns a time-shifted copy better than Euclidean", dtw_cost < eucl,
          f"dtw {dtw_cost:.3f} eucl {eucl:.3f}")

    # ---- 10. vector-valued series work --------------------------------------------------
    vx = [[0, 0], [1, 1], [2, 0], [3, 1]]
    vy = [[0, 0], [1, 1], [2, 0], [3, 1]]
    check("soft-DTW of identical vector series ~ 0", abs(S.soft_dtw(vx, vy, 0.001)) < 1e-2)
    check("hard DTW of identical vector series == 0", abs(S.hard_dtw(vx, vy)) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
