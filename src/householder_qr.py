"""Householder QR: orthogonal factorization by reflections.

The QR factorization writes a matrix A = Q R with Q orthogonal (its columns orthonormal) and R upper
triangular. It is the numerically stable backbone of least-squares regression, eigenvalue algorithms
(the QR iteration), and orthogonalization. Gram-Schmidt builds it column by column but loses
orthogonality to rounding on ill-conditioned inputs; the HOUSEHOLDER method builds it differently and is
the standard for its superior stability. Instead of orthogonalizing columns, it applies a sequence of
REFLECTIONS -- orthogonal transformations that mirror a vector across a hyperplane -- each chosen to
zero out everything below the diagonal in one column of A. After n reflections A has been triangularized
into R, and the product of the reflections (in reverse) is the orthogonal Q.

A single Householder reflection H = I - 2 v v^T / (v^T v) maps a target vector x onto a multiple of the
first basis vector: choose v = x - (-sign(x_1))||x|| e_1 (the sign chosen to avoid cancellation), and
H x becomes (+-||x||, 0, ..., 0). Applying H to the trailing submatrix zeros the sub-diagonal of the
current column; repeating on each successive column, using reflections acting on shrinking trailing
blocks, reduces A to upper-triangular R. Accumulating the reflections gives the full m x m orthogonal Q
(or the thin m x n Q). Because each reflection is exactly orthogonal, the computed Q stays orthonormal
to machine precision even when Gram-Schmidt would drift.

This module computes the Householder QR factorization (full and thin), and solves least-squares / square
linear systems with it. It is verified by reconstruction (Q R equals A), orthonormality (Q^T Q = I),
triangularity of R, and agreement of the least-squares solution with the normal-equations answer and
with the repository's Gram-Schmidt QR -- on hundreds of random matrices of assorted shapes. Pure stdlib;
a numerical-linear-algebra companion to the (Gram-Schmidt) QR, LU, and SVD notes."""

from __future__ import annotations

import math


def _matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for t in range(k):
            a = Ai[t]
            if a == 0:
                continue
            Bt = B[t]
            for j in range(m):
                Ci[j] += a * Bt[j]
    return C


def _transpose(A):
    return [list(row) for row in zip(*A)] if A else []


def qr(A):
    """Householder QR factorization of an m x n matrix A (m >= n). Returns (Q, R) with Q m x m
    orthogonal and R m x n upper triangular, Q R = A."""
    m = len(A)
    n = len(A[0]) if m else 0
    R = [row[:] for row in (float_row(r) for r in A)]
    Q = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

    for k in range(min(m, n)):
        # the column vector x = R[k:, k]
        x = [R[i][k] for i in range(k, m)]
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        if norm_x == 0:
            continue
        # v = x - alpha e1, alpha = -sign(x0)*||x|| (sign chosen to avoid cancellation)
        alpha = -norm_x if x[0] >= 0 else norm_x
        v = x[:]
        v[0] -= alpha
        vnorm2 = sum(vi * vi for vi in v)
        if vnorm2 == 0:
            continue
        # apply H = I - 2 v v^T / (v^T v) to the trailing submatrix R[k:, k:]
        for j in range(k, n):
            # dot = v . R[k:, j]
            dot = sum(v[i - k] * R[i][j] for i in range(k, m))
            factor = 2.0 * dot / vnorm2
            for i in range(k, m):
                R[i][j] -= factor * v[i - k]
        # accumulate into Q: Q = Q * H, applied to columns k.. of Q (rows here, since we build Q^T)
        for j in range(m):
            dot = sum(v[i - k] * Q[i][j] for i in range(k, m))
            factor = 2.0 * dot / vnorm2
            for i in range(k, m):
                Q[i][j] -= factor * v[i - k]

    # Q currently holds H_{n-1}...H_1 (an orthogonal matrix Q^T such that Q^T A = R); transpose it
    Qt = _transpose(Q)
    # clean tiny sub-diagonal noise in R
    for i in range(m):
        for j in range(min(i, n)):
            R[i][j] = 0.0
    return Qt, R


def float_row(r):
    return [float(x) for x in r]


def qr_thin(A):
    """Thin QR: the first n columns of Q (m x n) and the top n x n block of R. Q R = A."""
    Q, R = qr(A)
    m = len(A)
    n = len(A[0]) if m else 0
    Qthin = [[Q[i][j] for j in range(n)] for i in range(m)]
    Rthin = [[R[i][j] for j in range(n)] for i in range(n)]
    return Qthin, Rthin


def solve(A, b):
    """Solve the square system A x = b via Householder QR. Returns x."""
    Q, R = qr(A)
    n = len(A)
    # y = Q^T b
    Qt = _transpose(Q)
    y = [sum(Qt[i][j] * b[j] for j in range(n)) for i in range(n)]
    # back-substitution R x = y
    return _back_sub(R, y, n)


def lstsq(A, b):
    """Least-squares solution of the (possibly overdetermined) system A x ~= b via Householder QR."""
    Q, R = qr(A)
    m = len(A)
    n = len(A[0]) if m else 0
    Qt = _transpose(Q)
    y = [sum(Qt[i][j] * b[j] for j in range(m)) for i in range(n)]     # first n entries of Q^T b
    return _back_sub(R, y, n)


def _back_sub(R, y, n):
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i] - sum(R[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / R[i][i]
    return x


# --- validation helpers -----------------------------------------------------
def reconstruct(Q, R):
    """Q R as a matrix, for checking against A."""
    return _matmul(Q, R)


def is_orthonormal(Q, tol=1e-9):
    """True iff Q^T Q is the identity (columns orthonormal)."""
    n = len(Q[0]) if Q else 0
    for i in range(n):
        for j in range(n):
            dot = sum(Q[r][i] * Q[r][j] for r in range(len(Q)))
            expect = 1.0 if i == j else 0.0
            if abs(dot - expect) > tol:
                return False
    return True


def is_upper_triangular(R, tol=1e-9):
    for i in range(len(R)):
        for j in range(min(i, len(R[0]))):
            if abs(R[i][j]) > tol:
                return False
    return True
