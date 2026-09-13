"""Tests for exact distance transform: matches brute nearest-feature, 1-D envelope, edge cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distance_transform import (  # noqa: E402
    distance_transform_1d,
    squared_distance_transform_2d,
    distance_transform_2d,
    nearest_feature,
    brute_squared_distance_2d,
    brute_distance_1d,
    INF,
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


def main():
    # ---- 1. 1-D transform matches brute minimization -----------------------------------
    rng = _lcg(2024)
    mism = 0
    for _ in range(200):
        n = 1 + int(rng() * 20)
        # mix of 0 (feature) and inf (background)
        f = [0.0 if rng() < 0.3 else INF for _ in range(n)]
        if all(v == INF for v in f):
            f[int(rng() * n)] = 0.0  # ensure at least one feature
        d = distance_transform_1d(f)
        b = brute_distance_1d(f)
        if any(abs(d[i] - b[i]) > 1e-9 for i in range(n)):
            mism += 1
    check("1-D transform == brute minimization (200 arrays)", mism == 0, f"{mism}")

    # ---- 2. 2-D squared transform matches brute nearest-feature -------------------------
    rng = _lcg(77)
    mism = 0
    for _ in range(150):
        h = 1 + int(rng() * 8)
        w = 1 + int(rng() * 8)
        feature = [[rng() < 0.25 for _ in range(w)] for _ in range(h)]
        # ensure at least one feature
        if not any(any(row) for row in feature):
            feature[int(rng() * h)][int(rng() * w)] = True
        d = squared_distance_transform_2d(feature)
        b = brute_squared_distance_2d(feature)
        for i in range(h):
            for j in range(w):
                if abs(d[i][j] - b[i][j]) > 1e-9:
                    mism += 1
    check("2-D squared transform == brute nearest-feature (150 images)", mism == 0, f"{mism}")

    # ---- 3. feature pixels have distance zero -------------------------------------------
    feature = [[False, True, False], [False, False, False], [True, False, False]]
    d = distance_transform_2d(feature)
    check("feature pixels at distance 0", d[0][1] == 0 and d[2][0] == 0)

    # ---- 4. single feature point gives exact radial distances ---------------------------
    h = w = 7
    feature = [[False] * w for _ in range(h)]
    feature[3][3] = True
    d = distance_transform_2d(feature)
    ok = True
    for i in range(h):
        for j in range(w):
            expected = math.sqrt((i - 3) ** 2 + (j - 3) ** 2)
            if abs(d[i][j] - expected) > 1e-9:
                ok = False
    check("single point: exact radial distance", ok)

    # ---- 5. Euclidean transform is sqrt of squared transform ----------------------------
    feature = [[True, False, False, False, False]]
    sq = squared_distance_transform_2d(feature)
    eu = distance_transform_2d(feature)
    check("Euclidean = sqrt(squared)", all(abs(eu[0][j] - math.sqrt(sq[0][j])) < 1e-12
                                           for j in range(5)))
    check("row distances 0,1,2,3,4", [int(round(x)) for x in eu[0]] == [0, 1, 2, 3, 4])

    # ---- 6. nearest_feature returns the actual closest site -----------------------------
    feature = [[True, False, False], [False, False, False], [False, False, True]]
    dist, near = nearest_feature(feature)
    check("corner (0,0) nearest is itself", near[0][0] == (0, 0))
    check("corner (2,2) nearest is itself", near[2][2] == (2, 2))
    # center (1,1) is equidistant; nearest must be one of the two features at distance sqrt(2)
    check("center nearest at sqrt(2)", abs(dist[1][1] - math.sqrt(2)) < 1e-9)
    check("center nearest is a real feature", near[1][1] in [(0, 0), (2, 2)])

    # ---- 7. full-feature image is all zeros ---------------------------------------------
    feature = [[True, True], [True, True]]
    d = distance_transform_2d(feature)
    check("all-feature image: all distances 0", all(v == 0 for row in d for v in row))

    # ---- 8. all-background image is all INF ---------------------------------------------
    feature = [[False, False], [False, False]]
    sq = squared_distance_transform_2d(feature)
    check("all-background: all INF", all(v == INF for row in sq for v in row))

    # ---- 9. edge cases ------------------------------------------------------------------
    check("empty image", squared_distance_transform_2d([]) == [])
    check("1x1 feature", distance_transform_2d([[True]]) == [[0.0]])
    check("1-D single feature", distance_transform_1d([0.0]) == [0.0])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
