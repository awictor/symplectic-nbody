"""Tests for svd.py -- singular value decomposition and PCA.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Tolerances are ~1e-4 because the
SVD is built on the power-iteration eigensolver (not machine-precision).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import svd as S  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def mclose(A, B, tol=1e-4):
    return all(abs(A[i][j] - B[i][j]) <= tol for i in range(len(A)) for j in range(len(A[0])))


def orthonormal_cols(M, tol=1e-4):
    r = len(M[0])
    for a in range(r):
        for b in range(r):
            dot = sum(M[i][a] * M[i][b] for i in range(len(M)))
            if abs(dot - (1.0 if a == b else 0.0)) > tol:
                return False
    return True


# --- reconstruction A = U S V^T --------------------------------------------
A = [[3, 1, 1], [-1, 3, 1]]
U, s, Vt = S.svd(A)
check("A = U S V^T reconstructs the wide matrix", mclose(S.reconstruct(U, s, Vt), A))
check("U has orthonormal columns", orthonormal_cols(U))
check("V has orthonormal columns", orthonormal_cols(S._T(Vt)))
check("singular values are non-negative and descending",
      all(s[i] >= 0 for i in range(len(s))) and all(s[i] >= s[i + 1] - 1e-9 for i in range(len(s) - 1)))
# a tall matrix too
B = [[1, 0], [1, 1], [0, 1]]
Ub, sb, Vtb = S.svd(B)
check("A = U S V^T reconstructs a tall matrix", mclose(S.reconstruct(Ub, sb, Vtb), B))

# --- singular values of a diagonal matrix are its |entries| ----------------
D = [[3, 0, 0], [0, -5, 0], [0, 0, 1]]
_, sd, _ = S.svd(D)
check("singular values of a diagonal matrix are its sorted magnitudes",
      all(abs(a - b) < 1e-4 for a, b in zip(sorted(sd, reverse=True), [5, 3, 1])))

# --- rank -------------------------------------------------------------------
check("rank of a rank-1 matrix is 1", S.rank([[1, 2], [2, 4]]) == 1)
check("rank of the identity is 2", S.rank([[1, 0], [0, 1]]) == 2)
check("rank of a full-rank 3x3 is 3", S.rank([[2, 0, 1], [1, 3, 0], [0, 1, 4]]) == 3)
check("rank of a zero matrix is 0", S.rank([[0, 0], [0, 0]]) == 0)

# --- spectral norm and condition number ------------------------------------
check("||I||_2 = 1", abs(S.spectral_norm([[1, 0], [0, 1]]) - 1.0) < 1e-6)
check("spectral norm of diag(4,1) is 4", abs(S.spectral_norm([[4, 0], [0, 1]]) - 4.0) < 1e-4)
check("condition number of diag(4,1) is 4", abs(S.condition_number([[4, 0], [0, 1]]) - 4.0) < 1e-4)
check("condition number of a rank-deficient matrix is inf", S.condition_number([[1, 2], [2, 4]]) == float("inf"))

# --- low-rank approximation (Eckart-Young) ---------------------------------
M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]        # rank 2 (rows are collinear-ish)
check("full-rank approximation reconstructs the matrix", mclose(S.low_rank_approx(M, 3), M, 1e-3))
# the rank-1 approximation is the best rank-1 fit: its error <= rank-2 gap
approx1 = S.low_rank_approx(M, 1)
err1 = sum((approx1[i][j] - M[i][j]) ** 2 for i in range(3) for j in range(3)) ** 0.5
approx2 = S.low_rank_approx(M, 2)
err2 = sum((approx2[i][j] - M[i][j]) ** 2 for i in range(3) for j in range(3)) ** 0.5
check("higher-rank approximation has smaller error", err2 < err1 + 1e-9)
check("rank-2 approximation of a rank-2 matrix is near-exact", err2 < 1e-3)

# --- PCA on correlated 2D data ---------------------------------------------
# points on the line y = 2x: one component explains all the variance
data = [[float(i), 2.0 * i] for i in range(10)]
comps, var, mean = S.pca(data)
ratio = S.explained_variance_ratio(var)
check("PCA finds the dominant direction explains ~all variance", ratio[0] > 0.999)
check("the first component aligns with the data direction [1,2]/sqrt5",
      abs(abs(comps[0][0]) - 1 / math.sqrt(5)) < 1e-3 and abs(abs(comps[0][1]) - 2 / math.sqrt(5)) < 1e-3)
check("PCA mean is the column mean", abs(mean[0] - 4.5) < 1e-9)
# explained variances are descending
check("explained variances are descending", all(var[i] >= var[i + 1] - 1e-9 for i in range(len(var) - 1)))

# --- PCA projection reduces dimension ---------------------------------------
comps1, _, mean1 = S.pca(data, n_components=1)
scores = S.project(data, comps1, mean1)
check("projection onto 1 component gives 1D scores", all(len(row) == 1 for row in scores))
# the scores should be monotonic (data walks along the line)
flat = [row[0] for row in scores]
check("projected scores order the samples along the component",
      all(flat[i] <= flat[i + 1] for i in range(len(flat) - 1)) or all(flat[i] >= flat[i + 1] for i in range(len(flat) - 1)))

# --- explained-variance ratio sums to 1 ------------------------------------
data3 = [[math.sin(i), math.cos(i), math.sin(2 * i)] for i in range(20)]
_, var3, _ = S.pca(data3)
check("explained-variance ratios sum to 1", abs(sum(S.explained_variance_ratio(var3)) - 1.0) < 1e-9)
check("PCA of 3D data yields up to 3 components", len(var3) <= 3)

# --- a random matrix round-trips through its SVD ---------------------------
def lcg(seed):
    st = seed
    while True:
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        yield st >> 8


gen = lcg(1)
ok = True
for _ in range(20):
    m = 2 + next(gen) % 3
    n = 2 + next(gen) % 3
    R = [[(next(gen) / (1 << 24)) * 4 - 2 for _ in range(n)] for _ in range(m)]
    Ur, sr, Vtr = S.svd(R)
    if not mclose(S.reconstruct(Ur, sr, Vtr), R, 1e-3):
        ok = False
        break
check("random matrices reconstruct from their SVD", ok)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall svd tests passed")
