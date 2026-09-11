"""QR decomposition: orthogonalizing a matrix, and least squares.

Any matrix A (m rows, n columns, m >= n) factors as A = Q R, where Q has orthonormal columns
(Q^T Q = I) and R is upper-triangular. That factorization is the workhorse of overdetermined
systems: to solve the least-squares problem "find x minimizing ||A x - b||" -- fitting a model
to more data points than parameters -- you compute Q R once and back-substitute R x = Q^T b, far
more stable than forming the normal equations A^T A x = A^T b (which squares the condition
number).

The construction here is the GRAM-SCHMIDT process: sweep the columns left to right, subtract
from each the components already accounted for by the earlier orthonormal directions, and
normalize what remains. The naive ("classical") version loses orthogonality to rounding; the
MODIFIED Gram-Schmidt used here subtracts each projection immediately, which is far more stable.
R's entries are exactly the projection coefficients and the residual norms, so QR and
orthonormalization are the same computation.

QR also drives eigenvalue algorithms (the QR iteration), rank detection, and orthogonal
regression. This module builds the (thin) QR factorization by modified Gram-Schmidt, solves
least-squares and square systems, and checks Q^T Q = I, Q R = A, and that the least-squares
residual is orthogonal to the column space. Pure stdlib (lists of lists); the linear-algebra
companion to the LU (linsolve) note."""

from __future__ import annotations

import math


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _norm(v):
    return math.sqrt(_dot(v, v))


def _col(A, j):
    return [A[i][j] for i in range(len(A))]


def qr_decompose(A):
    """Thin QR factorization A = Q R by modified Gram-Schmidt. A is m x n (m >= n). Returns
    (Q, R) with Q an m x n matrix of orthonormal columns and R an n x n upper-triangular
    matrix. Raises if the columns are linearly dependent (rank-deficient)."""
    m = len(A)
    n = len(A[0])
    if m < n:
        raise ValueError("need m >= n (at least as many rows as columns)")
    # work on a mutable copy of the columns
    V = [_col(A, j) for j in range(n)]
    Qcols = [[0.0] * m for _ in range(n)]
    R = [[0.0] * n for _ in range(n)]
    for j in range(n):
        v = V[j]
        for i in range(j):
            R[i][j] = _dot(Qcols[i], v)          # projection of the (partially reduced) v
            v = [v[k] - R[i][j] * Qcols[i][k] for k in range(m)]
        R[j][j] = _norm(v)
        if R[j][j] < 1e-12:
            raise ValueError("rank-deficient matrix (linearly dependent columns)")
        Qcols[j] = [v[k] / R[j][j] for k in range(m)]
    # assemble Q as an m x n matrix (columns -> rows)
    Q = [[Qcols[j][i] for j in range(n)] for i in range(m)]
    return Q, R


def _matT_vec(Q, b):
    """Q^T b where Q is m x n: returns an n-vector."""
    m = len(Q)
    n = len(Q[0])
    return [sum(Q[i][j] * b[i] for i in range(m)) for j in range(n)]


def _back_substitution(R, y):
    n = len(R)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(R[i][j] * x[j] for j in range(i + 1, n))) / R[i][i]
    return x


def lstsq(A, b):
    """Least-squares solution of A x ~= b (m >= n): minimizes ||A x - b|| via QR, solving
    R x = Q^T b. For a square full-rank A this is the exact solution."""
    if len(b) != len(A):
        raise ValueError("dimension mismatch")
    Q, R = qr_decompose(A)
    return _back_substitution(R, _matT_vec(Q, b))


def solve(A, b):
    """Solve a square system A x = b via QR (an alternative to LU)."""
    if len(A) != len(A[0]):
        raise ValueError("solve needs a square matrix; use lstsq for rectangular A")
    return lstsq(A, b)


def is_orthonormal(Q, tol: float = 1e-9) -> bool:
    """True if Q's columns are orthonormal (Q^T Q = I)."""
    n = len(Q[0])
    for i in range(n):
        for j in range(n):
            dot = sum(Q[k][i] * Q[k][j] for k in range(len(Q)))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > tol:
                return False
    return True


def matmul(A, B):
    """Matrix product A B."""
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def residual(A, x, b):
    """The residual vector A x - b."""
    r = matvec(A, x)
    return [r[i] - b[i] for i in range(len(b))]


def residual_norm(A, x, b):
    """The Euclidean norm ||A x - b||."""
    return _norm(residual(A, x, b))


def reconstruct(Q, R):
    """Rebuild A = Q R from its factors (for checking)."""
    return matmul(Q, R)
