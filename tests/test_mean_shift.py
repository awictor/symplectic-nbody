"""Tests for mean shift: recovers blob count, mode locations, bandwidth effect, monotone ascent."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mean_shift import (  # noqa: E402
    mean_shift,
    climb,
    kde_value,
    n_clusters,
)


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
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _gauss(rng):
    u1 = max(rng(), 1e-12)
    u2 = rng()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _blobs(centers, n_each, spread, rng):
    pts = []
    for c in centers:
        for _ in range(n_each):
            pts.append([c[i] + spread * _gauss(rng) for i in range(len(c))])
    return pts


def main():
    # ---- 1. three well-separated blobs -> 3 clusters ------------------------------------
    rng = _lcg(2024)
    centers = [(0, 0), (10, 0), (5, 8)]
    pts = _blobs(centers, 20, 0.5, rng)
    labels, modes = mean_shift(pts, bandwidth=2.0, kernel="gaussian")
    check("three blobs -> 3 clusters", n_clusters(labels) == 3, f"{n_clusters(labels)}")
    check("3 modes recovered", len(modes) == 3, f"{len(modes)}")
    # each mode near a true centre
    def nearest_center(m):
        return min(centers, key=lambda c: (m[0] - c[0]) ** 2 + (m[1] - c[1]) ** 2)
    ok = all(math.dist(m, nearest_center(m)) < 1.0 for m in modes)
    check("modes near true blob centres", ok, f"{[tuple(round(x,1) for x in m) for m in modes]}")

    # ---- 2. two blobs -> 2 clusters -----------------------------------------------------
    rng = _lcg(77)
    pts = _blobs([(0, 0), (6, 6)], 25, 0.6, rng)
    labels, modes = mean_shift(pts, bandwidth=1.8)
    check("two blobs -> 2 clusters", n_clusters(labels) == 2, f"{n_clusters(labels)}")

    # ---- 3. single blob -> 1 cluster ----------------------------------------------------
    rng = _lcg(7)
    pts = _blobs([(3, 3)], 40, 0.7, rng)
    labels, modes = mean_shift(pts, bandwidth=2.0)
    check("single blob -> 1 cluster", n_clusters(labels) == 1, f"{n_clusters(labels)}")

    # ---- 4. bandwidth effect: smaller -> more clusters ----------------------------------
    rng = _lcg(555)
    pts = _blobs([(0, 0), (4, 0), (8, 0), (12, 0)], 20, 0.4, rng)
    small = n_clusters(mean_shift(pts, bandwidth=0.9)[0])
    large = n_clusters(mean_shift(pts, bandwidth=6.0)[0])
    check("smaller bandwidth finds more clusters", small >= large, f"{small} vs {large}")
    check("large bandwidth merges to few", large <= 2, f"{large}")

    # ---- 5. every point assigned to its nearest mode ------------------------------------
    rng = _lcg(11)
    pts = _blobs([(0, 0), (10, 10)], 15, 0.5, rng)
    labels, modes = mean_shift(pts, bandwidth=2.0)
    ok = True
    for i, p in enumerate(pts):
        nearest = min(range(len(modes)), key=lambda m: math.dist(p, modes[m]))
        if labels[i] != nearest:
            ok = False
    check("points labelled by nearest mode", ok)

    # ---- 6. mode-climb is a monotone density ascent -------------------------------------
    rng = _lcg(321)
    pts = _blobs([(0, 0), (8, 0)], 20, 0.6, rng)
    bw = 2.0
    # trace the climb manually and check KDE non-decreasing
    from mean_shift import _mean_shift_vector, _gaussian_kernel
    x = list(pts[0])
    prev_density = kde_value(x, pts, bw)
    ok = True
    for _ in range(50):
        x = _mean_shift_vector(x, pts, bw, _gaussian_kernel)
        d = kde_value(x, pts, bw)
        if d < prev_density - 1e-6:
            ok = False
        prev_density = d
    check("mode-climb monotonically increases density", ok)

    # ---- 7. climbed mode is a local density max -----------------------------------------
    mode = climb(pts[0], pts, bw)
    d_mode = kde_value(mode, pts, bw)
    # perturb the mode; density should not increase
    ok = True
    for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
        if kde_value([mode[0] + dx, mode[1] + dy], pts, bw) > d_mode + 1e-6:
            ok = False
    check("climbed point is a local density maximum", ok)

    # ---- 8. flat kernel also clusters ---------------------------------------------------
    rng = _lcg(13)
    pts = _blobs([(0, 0), (10, 0)], 20, 0.5, rng)
    labels, modes = mean_shift(pts, bandwidth=2.5, kernel="flat")
    check("flat kernel finds 2 clusters", n_clusters(labels) == 2, f"{n_clusters(labels)}")

    # ---- 9. edge cases ------------------------------------------------------------------
    labels, modes = mean_shift([[1.0, 1.0]], bandwidth=1.0)
    check("single point -> 1 cluster", n_clusters(labels) == 1)
    # 1-D data
    rng = _lcg(3)
    pts1d = _blobs([(0,), (10,)], 15, 0.4, rng)
    labels, _ = mean_shift(pts1d, bandwidth=2.0)
    check("1-D two blobs -> 2 clusters", n_clusters(labels) == 2, f"{n_clusters(labels)}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
