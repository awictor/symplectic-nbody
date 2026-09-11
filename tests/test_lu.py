"""Tests for lu: PA=LU and A=LL' reconstruction, solves, determinant, inverse, SPD test."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lu import (lu_decompose, lu_solve, determinant, inverse,
                cholesky, cholesky_solve, is_positive_definite)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def eq_mat(A, B, tol=1e-9):
    return all(approx(A[i][j], B[i][j], tol) for i in range(len(A)) for j in range(len(A[0])))


# --- LU reconstruction: P A = L U ------------------------------------------
A = [[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]]
L, U, piv, sign = lu_decompose(A)
PA = [[A[piv[i]][j] for j in range(3)] for i in range(3)]
check("P A = L U", eq_mat(PA, matmul(L, U)))
check("L is unit lower triangular",
      all(approx(L[i][i], 1.0) for i in range(3))
      and all(L[i][j] == 0.0 for i in range(3) for j in range(i + 1, 3)))
check("U is upper triangular", all(approx(U[i][j], 0.0) for i in range(3) for j in range(i)))

# --- solving A x = b -------------------------------------------------------
b = [5.0, -2.0, 9.0]
x = lu_solve(A, b)
residual = [sum(A[i][j] * x[j] for j in range(3)) - b[i] for i in range(3)]
check("lu_solve residual ~ 0", max(abs(r) for r in residual) < 1e-9)

# a 2x2 with a known exact solution: [[3,2],[1,2]] x = [7,5] -> x = [1,2]
x2 = lu_solve([[3.0, 2.0], [1.0, 2.0]], [7.0, 5.0])
check("lu_solve exact 2x2", approx(x2[0], 1.0) and approx(x2[1], 2.0))

# --- determinant vs cofactor reference -------------------------------------
check("det 2x2", approx(determinant([[1.0, 2.0], [3.0, 4.0]]), -2.0))
check("det 3x3 matches cofactor", approx(determinant(A), -16.0))
check("det identity is 1", approx(determinant([[1.0, 0.0], [0.0, 1.0]]), 1.0))
check("det singular is 0", approx(determinant([[1.0, 2.0], [2.0, 4.0]]), 0.0, 1e-9))
# determinant is multiplicative-ish: swapping two rows flips the sign
check("row swap flips det sign",
      approx(determinant([[0.0, 1.0], [1.0, 0.0]]), -1.0))

# --- inverse round-trips to the identity -----------------------------------
Ai = inverse(A)
check("A A^-1 = I", eq_mat(matmul(A, Ai), [[1.0 if i == j else 0.0 for j in range(3)]
                                           for i in range(3)]))
check("A^-1 A = I", eq_mat(matmul(Ai, A), [[1.0 if i == j else 0.0 for j in range(3)]
                                           for i in range(3)]))

# --- Cholesky: A = L L' for SPD --------------------------------------------
S = [[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]]
Lc = cholesky(S)
Lt = [[Lc[j][i] for j in range(3)] for i in range(3)]
check("L L' = S", eq_mat(matmul(Lc, Lt), S))
check("Cholesky L is lower triangular",
      all(approx(Lc[i][j], 0.0) for i in range(3) for j in range(i + 1, 3)))
check("Cholesky diagonal positive", all(Lc[i][i] > 0 for i in range(3)))

# a diagonal SPD matrix: L should be the elementwise sqrt
Ld = cholesky([[9.0, 0.0], [0.0, 16.0]])
check("Cholesky of diagonal", approx(Ld[0][0], 3.0) and approx(Ld[1][1], 4.0))

# --- Cholesky rejects non-positive-definite matrices -----------------------
def raises_valueerror(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


check("Cholesky rejects indefinite", raises_valueerror(lambda: cholesky([[1.0, 2.0], [2.0, 1.0]])))
check("Cholesky rejects negative diagonal",
      raises_valueerror(lambda: cholesky([[-1.0, 0.0], [0.0, 1.0]])))

# --- is_positive_definite --------------------------------------------------
check("SPD recognized", is_positive_definite(S))
check("indefinite rejected", not is_positive_definite([[1.0, 2.0], [2.0, 1.0]]))
check("asymmetric rejected", not is_positive_definite([[2.0, 1.0], [0.0, 2.0]]))

# --- Cholesky solve matches LU solve on SPD --------------------------------
bs = [1.0, 2.0, 3.0]
xc = cholesky_solve(S, bs)
xl = lu_solve(S, bs)
check("cholesky_solve matches lu_solve", all(approx(xc[i], xl[i]) for i in range(3)))
res_c = [sum(S[i][j] * xc[j] for j in range(3)) - bs[i] for i in range(3)]
check("cholesky_solve residual ~ 0", max(abs(r) for r in res_c) < 1e-9)

# --- a larger random SPD matrix (A = M M' + nI is always SPD) --------------
state = 123


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


n = 6
M = [[rng() * 2 - 1 for _ in range(n)] for _ in range(n)]
SPD = [[sum(M[i][k] * M[j][k] for k in range(n)) + (n if i == j else 0.0) for j in range(n)]
       for i in range(n)]
Lbig = cholesky(SPD)
Ltbig = [[Lbig[j][i] for j in range(n)] for i in range(n)]
check("large SPD reconstructs", eq_mat(matmul(Lbig, Ltbig), SPD, 1e-7))
bbig = [rng() for _ in range(n)]
xbig = cholesky_solve(SPD, bbig)
resbig = [sum(SPD[i][j] * xbig[j] for j in range(n)) - bbig[i] for i in range(n)]
check("large SPD solve residual ~ 0", max(abs(r) for r in resbig) < 1e-7)
check("large SPD is positive definite", is_positive_definite(SPD))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all lu tests passed")
