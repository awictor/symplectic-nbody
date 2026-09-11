"""Tests for mds: distance recovery, intrinsic dimension via eigenvalues, stress, Procrustes."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mds import (distance_matrix, classical_mds, stress, reconstructed_distances,
                 eigenvalue_spectrum, procrustes_align, _double_center)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


def max_dist_err(D, coords):
    n = len(D)
    Drec = reconstructed_distances(coords)
    return max(abs(D[i][j] - Drec[i][j]) for i in range(n) for j in range(n))


# --- distance matrix -------------------------------------------------------
sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
D = distance_matrix(sq)
check("distance matrix is symmetric", all(D[i][j] == D[j][i] for i in range(4) for j in range(4)))
check("zero diagonal", all(D[i][i] == 0.0 for i in range(4)))
check("unit square side length", approx(D[0][1], 1.0, 1e-12))
check("unit square diagonal", approx(D[0][2], math.sqrt(2), 1e-12))

# --- double centering yields a symmetric Gram matrix -----------------------
B = _double_center(D)
check("Gram matrix symmetric", all(approx(B[i][j], B[j][i], 1e-12) for i in range(4) for j in range(4)))

# --- recovers the square's distances exactly -------------------------------
coords, vals = classical_mds(D, 2)
check("square distances recovered", max_dist_err(D, coords) < 1e-8)
check("square stress ~ 0", stress(D, coords) < 1e-8)
check("square has two equal positive eigenvalues", approx(vals[0], vals[1], 1e-6) and vals[0] > 0)

# --- collinear points collapse to one dimension ---------------------------
line = [[float(i), 0.0] for i in range(5)]
spec = eigenvalue_spectrum(distance_matrix(line))
check("line has one dominant eigenvalue", spec[0] > 1.0)
check("line's remaining eigenvalues ~ 0", all(abs(v) < 1e-6 for v in spec[1:]))
# a 1-component MDS of the line reproduces its distances
c1, _ = classical_mds(distance_matrix(line), 1)
check("1-D MDS reproduces the line", stress(distance_matrix(line), c1) < 1e-8)

# --- a genuinely 2-D configuration has exactly two nonzero eigenvalues -----
pentagon = [[math.cos(2 * math.pi * k / 5), math.sin(2 * math.pi * k / 5)] for k in range(5)]
spec_p = eigenvalue_spectrum(distance_matrix(pentagon))
n_pos = sum(1 for v in spec_p if v > 1e-6)
check("2-D pentagon has two positive eigenvalues", n_pos == 2)

# --- random 3-D points: 3 nonzero eigenvalues, zero stress -----------------
state = 7


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


pts3 = [[rng() * 10, rng() * 10, rng() * 10] for _ in range(8)]
D3 = distance_matrix(pts3)
c3, v3 = classical_mds(D3, 3)
check("3-D distances recovered", max_dist_err(D3, c3) < 1e-6)
check("3-D stress ~ 0", stress(D3, c3) < 1e-6)
check("3-D config has three positive eigenvalues", all(v > 1e-6 for v in v3))
spec3 = eigenvalue_spectrum(D3)
check("4th eigenvalue of 3-D data ~ 0", abs(spec3[3]) < 1e-6)

# --- higher-component request pads flat dimensions with zeros --------------
c_over, v_over = classical_mds(D, 3)   # square is 2-D, ask for 3
check("over-request keeps shape", len(c_over[0]) == 3)
check("extra eigenvalue is ~0", approx(v_over[2], 0.0, 1e-6))
check("padded coordinate is zero", all(approx(row[2], 0.0, 1e-9) for row in c_over))

# --- distances are preserved regardless of embedding dimension chosen ------
c2_ok, _ = classical_mds(D3, 3)
check("reconstruction preserves all pairwise distances", max_dist_err(D3, c2_ok) < 1e-6)

# --- Procrustes aligns a reconstruction back onto the known configuration --
c_align, _ = classical_mds(D, 2)
aligned = procrustes_align(sq, c_align)
check("Procrustes recovers the original square",
      max(abs(sq[i][k] - aligned[i][k]) for i in range(4) for k in range(2)) < 1e-6)
# Procrustes preserves internal distances (it is a rigid motion)
check("Procrustes preserves shape", max_dist_err(D, aligned) < 1e-6)

# --- MDS on a known 2-D set, aligned, matches the truth --------------------
truth = [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0], [1.5, 2.0]]
Dt = distance_matrix(truth)
rec, _ = classical_mds(Dt, 2)
rec_aligned = procrustes_align(truth, rec)
check("recovers a rectangle+center up to rigid motion",
      max(abs(truth[i][k] - rec_aligned[i][k]) for i in range(5) for k in range(2)) < 1e-5)

# --- empty input -----------------------------------------------------------
check("empty distance matrix -> empty", classical_mds([], 2) == ([], []))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all mds tests passed")
