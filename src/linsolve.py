"""Gaussian elimination and LU decomposition: solving linear systems.

A x = b -- n equations in n unknowns -- is the most-solved problem in computation: circuit
analysis, structural mechanics, least-squares fitting, the linearized step of every nonlinear
solver. Gaussian elimination row-reduces the augmented matrix to triangular form and
back-substitutes, in O(n^3). Done carefully, once, it factors A = L U into a lower- and an
upper-triangular matrix, after which EACH new right-hand side is solved in O(n^2) by two
triangular sweeps -- so you factor once and reuse.

The one subtlety is stability. A small or zero pivot on the diagonal blows up the elimination;
PARTIAL PIVOTING swaps in the row with the largest pivot at each step, which keeps the
multipliers bounded and makes the method numerically robust. The pivot swaps are recorded as a
permutation P, giving P A = L U. From the factorization two quantities fall out for free: the
DETERMINANT is the product of U's diagonal (times the sign of the permutation), and the INVERSE
comes from solving A x = e_i for each unit column.

This module builds the LU factorization with partial pivoting, solves systems, computes
determinants and inverses, and checks everything against known solutions, residuals ||Ax - b||,
and the identity P A = L U. Pure stdlib (lists of lists); the linear-algebra companion to the
FFT and quadrature notes."""

from __future__ import annotations


def lu_decompose(A):
    """LU factorization with partial pivoting: returns (L, U, perm, sign) with P A = L U, where
    perm is the row permutation (as an index list) and sign is +-1 (the permutation's parity,
    for the determinant). L is unit-lower-triangular, U upper-triangular."""
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("matrix must be square")
    U = [list(map(float, row)) for row in A]
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    perm = list(range(n))
    sign = 1
    for k in range(n):
        # partial pivot: largest-magnitude entry in column k at or below the diagonal
        pivot = max(range(k, n), key=lambda i: abs(U[i][k]))
        if abs(U[pivot][k]) < 1e-15:
            raise ValueError("matrix is singular (zero pivot)")
        if pivot != k:
            U[k], U[pivot] = U[pivot], U[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]
            sign = -sign
            for j in range(k):                 # swap the already-computed part of L
                L[k][j], L[pivot][j] = L[pivot][j], L[k][j]
        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]
            L[i][k] = m
            for j in range(k, n):
                U[i][j] -= m * U[k][j]
    return L, U, perm, sign


def _forward_substitution(L, b):
    """Solve L y = b for unit-lower-triangular L."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y


def _back_substitution(U, y):
    """Solve U x = y for upper-triangular U."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def solve(A, b):
    """Solve A x = b by LU with partial pivoting. Returns the solution vector x."""
    if len(b) != len(A):
        raise ValueError("dimension mismatch")
    L, U, perm, _ = lu_decompose(A)
    pb = [b[perm[i]] for i in range(len(b))]    # apply the row permutation to b
    y = _forward_substitution(L, pb)
    return _back_substitution(U, y)


def solve_lu(L, U, perm, b):
    """Solve A x = b reusing a precomputed factorization (each new b costs O(n^2))."""
    pb = [b[perm[i]] for i in range(len(b))]
    return _back_substitution(U, _forward_substitution(L, pb))


def determinant(A):
    """Determinant of A via its LU factorization: sign * product of U's diagonal. Returns 0 for
    a singular matrix."""
    try:
        _, U, _, sign = lu_decompose(A)
    except ValueError:
        return 0.0
    det = float(sign)
    for i in range(len(U)):
        det *= U[i][i]
    return det


def inverse(A):
    """Inverse of A by solving A x = e_i for each unit column, reusing one factorization."""
    n = len(A)
    L, U, perm, _ = lu_decompose(A)
    cols = []
    for i in range(n):
        e = [1.0 if j == i else 0.0 for j in range(n)]
        cols.append(solve_lu(L, U, perm, e))
    # cols[i] is the i-th column of the inverse; transpose into rows
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def matvec(A, x):
    """Matrix-vector product A x."""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def matmul(A, B):
    """Matrix product A B."""
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def residual_norm(A, x, b):
    """The Euclidean norm ||A x - b|| -- a direct measure of how well x solves the system."""
    r = matvec(A, x)
    return sum((r[i] - b[i]) ** 2 for i in range(len(b))) ** 0.5


def identity(n):
    """The n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
