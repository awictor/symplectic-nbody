"""Tests for householder_qr: reflection QR by reconstruction, orthonormality, and solves."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from householder_qr import (qr, qr_thin, solve, lstsq, reconstruct, is_orthonormal,
                            is_upper_triangular)
from qr import qr_decompose, lstsq as gs_lstsq   # repository Gram-Schmidt QR, for cross-check

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
    if len(A) != len(B):
        return False
    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            return False
        for j in range(len(A[i])):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


def random_matrix(rng, m, n):
    return [[rng.randf(-6, 6) for _ in range(n)] for _ in range(m)]


# --- known matrix -----------------------------------------------------------
A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
Q, R = qr(A)
check("QR reconstructs the classic matrix", mat_close(reconstruct(Q, R), [list(map(float, r)) for r in A]))
check("Q is orthonormal", is_orthonormal(Q))
check("R is upper triangular", is_upper_triangular(R))

# --- reconstruction + orthonormality on random square/tall matrices --------
rng = LCG(2026)
recon_ok = orth_ok = tri_ok = True
for _ in range(300):
    m = rng.rand() % 6 + 2
    n = rng.rand() % m + 1              # n <= m
    A = random_matrix(rng, m, n)
    Af = [[float(x) for x in row] for row in A]
    Q, R = qr(A)
    if not mat_close(reconstruct(Q, R), Af):
        recon_ok = False
        break
    if not is_orthonormal(Q):
        orth_ok = False
        break
    if not is_upper_triangular(R):
        tri_ok = False
        break
check("Q R reconstructs A on random matrices (300)", recon_ok)
check("Q stays orthonormal on random matrices", orth_ok)
check("R stays upper triangular on random matrices", tri_ok)

# --- Q is genuinely orthogonal: Q^T Q = I (full m x m) ---------------------
rng = LCG(4242)
full_orth_ok = True
for _ in range(200):
    m = rng.rand() % 5 + 2
    n = rng.rand() % m + 1
    A = random_matrix(rng, m, n)
    Q, R = qr(A)
    # Q is m x m
    if len(Q) != m or len(Q[0]) != m:
        full_orth_ok = False
        break
    if not is_orthonormal(Q):
        full_orth_ok = False
        break
check("the full Q is m x m and orthonormal", full_orth_ok)

# --- thin QR reconstructs too ----------------------------------------------
rng = LCG(777)
thin_ok = True
for _ in range(200):
    m = rng.rand() % 6 + 2
    n = rng.rand() % m + 1
    A = random_matrix(rng, m, n)
    Af = [[float(x) for x in row] for row in A]
    Q, R = qr_thin(A)
    if len(Q[0]) != n or len(R) != n:
        thin_ok = False
        break
    if not mat_close(reconstruct(Q, R), Af):
        thin_ok = False
        break
check("thin QR (m x n Q, n x n R) reconstructs A", thin_ok)

# --- square solve ----------------------------------------------------------
rng = LCG(555)
solve_ok = True
for _ in range(300):
    n = rng.rand() % 5 + 1
    A = random_matrix(rng, n, n)
    # make well-conditioned by adding to the diagonal
    for i in range(n):
        A[i][i] += 10
    x_true = [rng.randf(-5, 5) for _ in range(n)]
    b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]
    x = solve(A, b)
    if any(abs(x[i] - x_true[i]) > 1e-6 for i in range(n)):
        solve_ok = False
        break
check("square solve recovers the true solution (300 systems)", solve_ok)

# --- least squares matches the normal equations ----------------------------
def normal_equations_lstsq(A, b):
    m = len(A)
    n = len(A[0])
    # AtA x = At b, solved by Gaussian elimination
    AtA = [[sum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    Atb = [sum(A[k][i] * b[k] for k in range(m)) for i in range(n)]
    # solve AtA x = Atb
    M = [row[:] + [Atb[i]] for i, row in enumerate(AtA)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        pivval = M[c][c]
        M[c] = [v / pivval for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [M[r][t] - f * M[c][t] for t in range(n + 1)]
    return [M[i][n] for i in range(n)]


rng = LCG(31337)
lstsq_ok = True
for _ in range(200):
    m = rng.rand() % 6 + 3
    n = rng.rand() % (m - 1) + 1          # overdetermined
    A = random_matrix(rng, m, n)
    b = [rng.randf(-5, 5) for _ in range(m)]
    hh = lstsq(A, b)
    ne = normal_equations_lstsq(A, b)
    if any(abs(hh[i] - ne[i]) > 1e-5 for i in range(n)):
        lstsq_ok = False
        break
check("least-squares solution matches the normal equations (200 overdetermined)", lstsq_ok)

# --- agreement with the repository Gram-Schmidt QR least squares -----------
rng = LCG(99)
gs_ok = True
for _ in range(150):
    m = rng.rand() % 6 + 3
    n = rng.rand() % (m - 1) + 1
    A = random_matrix(rng, m, n)
    b = [rng.randf(-5, 5) for _ in range(m)]
    hh = lstsq(A, b)
    gs = gs_lstsq(A, b)
    if any(abs(hh[i] - gs[i]) > 1e-5 for i in range(n)):
        gs_ok = False
        break
check("Householder least squares agrees with Gram-Schmidt QR least squares", gs_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all householder_qr tests passed")
