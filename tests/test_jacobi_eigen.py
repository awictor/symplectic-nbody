"""Tests for jacobi_eigen: symmetric eigendecomposition by reconstruction, identities, known spectra."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jacobi_eigen import (eigen, sorted_eigen, reconstruct, is_orthogonal, matvec)
from eigen import eigenvalues_symmetric   # repository power-iteration eigenvalues, for cross-check

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randf(self, lo, hi):
        return lo + (self.rand() / 65536.0) * (hi - lo)


def mat_close(A, B, tol=1e-7):
    return all(abs(A[i][j] - B[i][j]) <= tol for i in range(len(A)) for j in range(len(A[0])))


def random_symmetric(rng, n):
    M = [[rng.randf(-5, 5) for _ in range(n)] for _ in range(n)]
    return [[(M[i][j] + M[j][i]) / 2 for j in range(n)] for i in range(n)]


def det(A):
    # small-matrix determinant by Gaussian elimination
    n = len(A)
    a = [row[:] for row in A]
    d = 1.0
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-15:
            return 0.0
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
            d = -d
        d *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [a[r][t] - f * a[c][t] for t in range(n)]
    return d


# --- known spectra ----------------------------------------------------------
vals, V = eigen([[2, 1], [1, 2]])
check("[[2,1],[1,2]] has eigenvalues {1,3}", sorted(round(v, 6) for v in vals) == [1.0, 3.0])
check("its eigenvectors are orthogonal", is_orthogonal(V))
check("its reconstruction equals A", mat_close(reconstruct(vals, V), [[2.0, 1.0], [1.0, 2.0]]))

# diagonal matrix: eigenvalues are the diagonal
D = [[3, 0, 0], [0, -1, 0], [0, 0, 5]]
vals, V = eigen(D)
check("diagonal matrix: eigenvalues are its diagonal", sorted(round(v, 6) for v in vals) == [-1.0, 3.0, 5.0])

# identity: all eigenvalues 1
vals, _ = eigen([[1, 0], [0, 1]])
check("identity: all eigenvalues 1", all(abs(v - 1) < 1e-9 for v in vals))

# 1x1
vals, V = eigen([[7]])
check("1x1 matrix", vals == [7.0] and V == [[1.0]])

# --- reconstruction, orthogonality, and eigen-equation on random matrices --
rng = LCG(2026)
recon_ok = orth_ok = eq_ok = True
for _ in range(300):
    n = rng.rand() % 5 + 1
    A = random_symmetric(rng, n)
    vals, V = eigen(A)
    if not mat_close(reconstruct(vals, V), A, 1e-6):
        recon_ok = False
        break
    if not is_orthogonal(V, 1e-7):
        orth_ok = False
        break
    for c in range(n):
        v = [V[r][c] for r in range(n)]
        Av = matvec(A, v)
        for i in range(n):
            if abs(Av[i] - vals[c] * v[i]) > 1e-6:
                eq_ok = False
                break
        if not eq_ok:
            break
    if not eq_ok:
        break
check("V D V^T reconstructs A on random symmetric matrices (300)", recon_ok)
check("eigenvectors are orthonormal (V orthogonal)", orth_ok)
check("each eigenpair satisfies A v = lambda v", eq_ok)

# --- trace and determinant identities --------------------------------------
rng = LCG(4242)
trace_ok = det_ok = True
for _ in range(300):
    n = rng.rand() % 5 + 1
    A = random_symmetric(rng, n)
    vals, _ = eigen(A)
    trace = sum(A[i][i] for i in range(n))
    if abs(sum(vals) - trace) > 1e-6:
        trace_ok = False
        break
    prod = 1.0
    for v in vals:
        prod *= v
    if abs(prod - det(A)) > 1e-4 * max(1.0, abs(det(A))):
        det_ok = False
        break
check("sum of eigenvalues equals the trace (300 matrices)", trace_ok)
check("product of eigenvalues equals the determinant (300 matrices)", det_ok)

# --- sorted_eigen is descending --------------------------------------------
rng = LCG(777)
sort_ok = True
for _ in range(200):
    n = rng.rand() % 5 + 1
    A = random_symmetric(rng, n)
    vals, V = sorted_eigen(A)
    if any(vals[i] < vals[i + 1] - 1e-12 for i in range(len(vals) - 1)):
        sort_ok = False
        break
    if not mat_close(reconstruct(vals, V), A, 1e-6):
        sort_ok = False
        break
check("sorted_eigen returns descending eigenvalues and still reconstructs A", sort_ok)

# --- agreement with the repository power-iteration eigenvalues -------------
rng = LCG(31337)
agree_ok = True
for _ in range(150):
    n = rng.rand() % 4 + 2
    A = random_symmetric(rng, n)
    jac = sorted(eigen(A)[0], reverse=True)
    # power-iteration returns largest-magnitude first; compare as sorted-by-value sets
    pw = sorted(eigenvalues_symmetric(A)[0], reverse=True)
    if len(jac) != len(pw):
        agree_ok = False
        break
    if any(abs(jac[i] - pw[i]) > 1e-4 for i in range(len(jac))):
        agree_ok = False
        break
check("Jacobi eigenvalues agree with the power-iteration eigenvalues (150 matrices)", agree_ok)

# --- a larger matrix solves and reconstructs -------------------------------
rng = LCG(99)
A = random_symmetric(rng, 12)
vals, V = eigen(A)
check("12x12 symmetric matrix reconstructs from its Jacobi eigendecomposition",
      mat_close(reconstruct(vals, V), A, 1e-5) and is_orthogonal(V, 1e-6))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all jacobi_eigen tests passed")
