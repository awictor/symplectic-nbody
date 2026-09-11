"""Tests for dtw: identity, symmetry, stretch invariance, path validity, band, vs Euclidean."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dtw import dtw, dtw_path, dtw_multi, euclidean_distance, _abs_dist

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 314
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_dtw(x, y):
    """Reference DP DTW (no band), independent implementation."""
    n, m = len(x), len(y)
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(x[i - 1] - y[j - 1])
            D[i][j] = c + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    return D[n][m]


# --- identity and zero -----------------------------------------------------
check("identical series have zero distance", dtw([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0)
check("single-point identical", dtw([5], [5]) == 0.0)
check("empty vs empty is zero", dtw([], []) == 0.0)

# --- symmetry --------------------------------------------------------------
ok = True
for _ in range(50):
    a = [int(rng() * 10) for _ in range(1 + int(rng() * 8))]
    b = [int(rng() * 10) for _ in range(1 + int(rng() * 8))]
    if abs(dtw(a, b) - dtw(b, a)) > 1e-9:
        ok = False
        break
check("DTW is symmetric", ok)

# --- matches an independent brute-force DP ---------------------------------
ok = True
for _ in range(100):
    a = [int(rng() * 20) for _ in range(1 + int(rng() * 10))]
    b = [int(rng() * 20) for _ in range(1 + int(rng() * 10))]
    if abs(dtw(a, b) - brute_dtw(a, b)) > 1e-9:
        ok = False
        break
check("DTW matches an independent DP over 100 random pairs", ok)

# --- time-stretch invariance -----------------------------------------------
base = [1, 3, 2, 5, 4]
stretched = []
for v in base:
    stretched += [v, v, v]      # triple every point
check("duplicating points doesn't change the distance", dtw(base, stretched) == 0.0)

# a non-uniform stretch
nonuniform = [1, 1, 3, 2, 2, 2, 5, 4, 4]
check("non-uniform stretch of the same shape is still zero", dtw(base, nonuniform) == 0.0)

# --- DTW beats Euclidean on a shifted signal -------------------------------
a = [0, 1, 2, 3, 2, 1, 0, 0]
b = [0, 0, 1, 2, 3, 2, 1, 0]      # a shifted right by one
check("DTW distance <= Euclidean on a shifted signal", dtw(a, b) <= euclidean_distance(a, b))
check("DTW is much smaller for the shifted signal", dtw(a, b) < euclidean_distance(a, b))

# --- warping path validity -------------------------------------------------
d, path = dtw_path([1, 2, 3, 4], [1, 2, 2, 3, 4])
check("path starts at (0,0)", path[0] == (0, 0))
check("path ends at the corners", path[-1] == (3, 4))
# monotone non-decreasing in both indices, steps of at most 1
mono = all(path[k + 1][0] >= path[k][0] and path[k + 1][1] >= path[k][1]
           and (path[k + 1][0] - path[k][0]) <= 1 and (path[k + 1][1] - path[k][1]) <= 1
           for k in range(len(path) - 1))
check("warping path is monotone with unit steps", mono)
# the path cost equals the DTW distance
path_cost = sum(_abs_dist([1, 2, 3, 4][i], [1, 2, 2, 3, 4][j]) for i, j in path)
check("summed path cost equals the DTW distance", abs(path_cost - d) < 1e-9)

# --- a hand-computed small grid --------------------------------------------
# x=[1,2], y=[2,3]: D grid -> DTW should be |1-2| + |2-3| aligned... compute:
# best path (0,0)->(1,1): |1-2| + |2-3| = 1 + 1 = 2
check("hand-computed 2x2 DTW is 2", dtw([1, 2], [2, 3]) == 2.0)

# --- Sakoe-Chiba band ------------------------------------------------------
# with a wide band, result equals unconstrained DTW
a = [int(rng() * 10) for _ in range(12)]
b = [int(rng() * 10) for _ in range(12)]
check("wide band equals unconstrained DTW", abs(dtw(a, b, band=12) - dtw(a, b)) < 1e-9)
# a band never gives a smaller distance than unconstrained (it restricts the path space)
check("banded DTW >= unconstrained DTW", dtw(a, b, band=2) >= dtw(a, b) - 1e-9)

# --- multi-dimensional -----------------------------------------------------
p = [(0, 0), (1, 1), (2, 2)]
q = [(0, 0), (1, 1), (1, 1), (2, 2)]     # stretched
check("multi-dim DTW handles stretched vector series", dtw_multi(p, q) == 0.0)
r = [(0, 0), (3, 4)]
check("multi-dim uses Euclidean local distance", abs(dtw_multi([(0, 0)], [(3, 4)]) - 5.0) < 1e-9)

# --- non-negativity --------------------------------------------------------
ok = all(dtw([int(rng() * 10) for _ in range(5)], [int(rng() * 10) for _ in range(5)]) >= 0
         for _ in range(30))
check("DTW distance is always non-negative", ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all dtw tests passed")
