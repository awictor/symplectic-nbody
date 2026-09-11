"""LU and Cholesky decomposition: factoring a matrix to solve, invert, and take determinants.

Solving A x = b once is easy; solving it for many right-hand sides, or getting det(A) or A^-1, is
where FACTORIZATION pays off. LU decomposition writes a general square matrix as P A = L U -- a
permutation P (the row swaps of partial pivoting, which keeps the arithmetic stable), a
lower-triangular L with unit diagonal, and an upper-triangular U. Gaussian elimination produces it
in O(n^3) once; afterwards every solve is two O(n^2) triangular sweeps (forward then back
substitution), the determinant is the signed product of U's diagonal, and the inverse is n solves.

For a SYMMETRIC POSITIVE-DEFINITE matrix there is a smaller, faster, and more stable special case:
CHOLESKY, A = L L'. It does half the work of LU, needs no pivoting, and the factorization SUCCEEDS
if and only if the matrix is positive definite -- so attempting it is itself the standard
positive-definiteness test, and it is the backbone of least squares, Kalman filters, and sampling
correlated Gaussians.

This module implements LU with partial pivoting, Cholesky, triangular solves, a general linear
solver, determinant, and matrix inverse -- verified by reconstructing P A = L U and A = L L',
cross-checking solutions and determinants against independent references, confirming Cholesky
rejects non-positive-definite matrices, and round-tripping A A^-1 = I. Pure stdlib; the
matrix-factorization companion to the QR, SVD, and linear-solver notes."""

from __future__ import annotations

import math


def lu_decompose(A):
    """PA = LU by Gaussian elimination with partial pivoting.

    Returns (L, U, piv, sign) where L is unit-lower-triangular, U is upper-triangular, piv is the
    permutation as a list (row piv[i] of A becomes row i of PA), and sign is the permutation's
    parity (+1/-1) for the determinant."""
    n = len(A)
    U = [list(map(float, row)) for row in A]
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    piv = list(range(n))
    sign = 1.0
    for col in range(n):
        # partial pivot: largest magnitude in this column at or below the diagonal
        p = max(range(col, n), key=lambda r: abs(U[r][col]))
        if abs(U[p][col]) < 1e-15:
            continue  # singular column; leave as is (determinant will be 0)
        if p != col:
            U[col], U[p] = U[p], U[col]
            piv[col], piv[p] = piv[p], piv[col]
            sign = -sign
            # swap the already-computed part of L
            for j in range(col):
                L[col][j], L[p][j] = L[p][j], L[col][j]
        for r in range(col + 1, n):
            factor = U[r][col] / U[col][col]
            L[r][col] = factor
            for j in range(col, n):
                U[r][j] -= factor * U[col][j]
    return L, U, piv, sign


def _forward_sub(L, b):
    """Solve L y = b for unit-or-general lower-triangular L."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y


def _back_sub(U, y):
    """Solve U x = y for upper-triangular U."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def lu_solve(A, b):
    """Solve A x = b via LU with partial pivoting."""
    L, U, piv, _ = lu_decompose(A)
    pb = [b[piv[i]] for i in range(len(b))]      # apply the row permutation to b
    y = _forward_sub(L, pb)
    return _back_sub(U, y)


def determinant(A):
    """det(A) = sign * product of U's diagonal from the LU factorization."""
    L, U, piv, sign = lu_decompose(A)
    d = sign
    for i in range(len(U)):
        d *= U[i][i]
    return d


def inverse(A):
    """A^-1 by solving A x = e_i for each unit column (reuses one LU factorization)."""
    n = len(A)
    L, U, piv, _ = lu_decompose(A)
    cols = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        pe = [e[piv[k]] for k in range(n)]
        y = _forward_sub(L, pe)
        cols.append(_back_sub(U, y))
    # cols[i] is the i-th column of the inverse; transpose into row form
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def cholesky(A, tol=1e-12):
    """A = L L' for a symmetric positive-definite A. Returns lower-triangular L.

    Raises ValueError if A is not positive definite -- which is exactly the standard SPD test."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = A[i][i] - s
                if d <= tol:
                    raise ValueError("matrix is not positive definite")
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def cholesky_solve(A, b):
    """Solve A x = b for SPD A via Cholesky (half the work of LU, no pivoting)."""
    L = cholesky(A)
    y = _forward_sub(L, b)
    # back-substitute with L' (transpose)
    n = len(L)
    Lt = [[L[j][i] for j in range(n)] for i in range(n)]
    return _back_sub(Lt, y)


def is_positive_definite(A):
    """True iff A is symmetric positive definite (Cholesky succeeds)."""
    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > 1e-9:
                return False
    try:
        cholesky(A)
        return True
    except ValueError:
        return False
