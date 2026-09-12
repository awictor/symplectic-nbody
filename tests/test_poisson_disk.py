"""Tests for poisson_disk: min-distance invariant, maximality, density bounds, reproducibility."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poisson_disk import (poisson_disk_2d, poisson_disk_nd, dart_throwing_2d,  # noqa: E402
                          min_pairwise_distance, is_maximal_2d, gap_fraction_2d, _dist_sq)


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
    W, H, R = 100.0, 100.0, 6.0

    # ---- 1. minimum-distance invariant (the defining property) ------------------------
    for seed in (1, 7, 42, 100, 2024):
        pts = poisson_disk_2d(W, H, R, seed=seed)
        mind = min_pairwise_distance(pts)
        check(f"no two points closer than r (seed {seed})", mind >= R - 1e-9,
              f"min dist {mind:.4f} < r={R}")

    # ---- 2. all points inside the domain ----------------------------------------------
    pts = poisson_disk_2d(W, H, R, seed=42)
    inside = all(0 <= x < W and 0 <= y < H for x, y in pts)
    check("all samples lie inside the domain", inside)

    # ---- 3. near-maximality: almost no room left for another point --------------------
    # Bridson with finite k is near-maximal; the insertable fraction of the domain is tiny.
    gap = gap_fraction_2d(pts, W, H, R)
    check("packing is near-maximal (insertable gap < 1%)", gap < 0.01, f"gap fraction {gap:.4f}")
    # a higher candidate count k drives the gap to (essentially) zero
    dense = poisson_disk_2d(W, H, R, k=120, seed=42)
    check("higher k gives a strictly maximal packing", is_maximal_2d(dense, W, H, R, tol=0.0),
          f"gap {gap_fraction_2d(dense, W, H, R):.4f}")

    # ---- 4. density bounds -------------------------------------------------------------
    # Each point "owns" at least a disk of radius r/2 (area pi r^2/4) and packing can't exceed
    # the hexagonal bound. Loose sanity window:
    area = W * H
    n = len(pts)
    # upper: hexagonal close packing density ~0.9069 of disks radius r/2
    max_pts = 0.9069 * area / (math.pi * (R / 2) ** 2)
    # lower: a maximal set covers the plane with disks radius r, so n >= area / (pi r^2)
    min_pts = area / (math.pi * R ** 2)
    check("point count within packing density bounds",
          min_pts <= n <= max_pts * 1.05,
          f"n={n}, bounds [{min_pts:.1f}, {max_pts:.1f}]")

    # ---- 5. reproducibility ------------------------------------------------------------
    a = poisson_disk_2d(W, H, R, seed=555)
    b = poisson_disk_2d(W, H, R, seed=555)
    check("same seed -> identical sample set", a == b)
    c = poisson_disk_2d(W, H, R, seed=556)
    check("different seed -> different sample set", a != c)

    # ---- 6. smaller radius -> more points ---------------------------------------------
    few = poisson_disk_2d(W, H, 12.0, seed=3)
    many = poisson_disk_2d(W, H, 5.0, seed=3)
    check("smaller radius yields more points", len(many) > len(few),
          f"{len(many)} vs {len(few)}")

    # ---- 7. dart-throwing reference also respects the invariant, comparable density ----
    dart = dart_throwing_2d(50.0, 50.0, R, max_attempts=3000, seed=9)
    dmind = min_pairwise_distance(dart)
    check("dart-throwing reference respects min distance", dmind >= R - 1e-9,
          f"min dist {dmind:.4f}")
    grid_small = poisson_disk_2d(50.0, 50.0, R, seed=9)
    # Bridson should be at least as dense as a dart-thrower that gives up early
    check("Bridson at least as dense as early-stopping dart thrower",
          len(grid_small) >= 0.8 * len(dart), f"bridson={len(grid_small)} dart={len(dart)}")

    # ---- 8. arbitrary dimension: 3D min-distance invariant ----------------------------
    pts3 = poisson_disk_nd((40.0, 40.0, 40.0), 8.0, seed=11)
    check("3D sampler produces multiple points", len(pts3) > 5, f"n={len(pts3)}")
    mind3 = min_pairwise_distance(pts3)
    check("3D min-distance invariant holds", mind3 >= 8.0 - 1e-9, f"min dist {mind3:.4f}")
    inside3 = all(all(0 <= p[d] < 40.0 for d in range(3)) for p in pts3)
    check("3D points inside the box", inside3)

    # ---- 9. 1D reduces to evenly-ish spaced points ------------------------------------
    pts1 = poisson_disk_nd((100.0,), 10.0, seed=4)
    mind1 = min_pairwise_distance(pts1)
    check("1D min-distance invariant holds", mind1 >= 10.0 - 1e-9, f"{mind1:.4f}")
    check("1D produces a reasonable count", 5 <= len(pts1) <= 11, f"n={len(pts1)}")

    # ---- 10. invalid radius -----------------------------------------------------------
    try:
        poisson_disk_2d(W, H, 0.0)
        check("zero radius raises", False)
    except ValueError:
        check("zero radius raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
