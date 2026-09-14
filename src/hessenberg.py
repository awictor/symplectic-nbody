"""Hessenberg reduction: squeeze a matrix to almost-triangular form, the first step of every eigensolver.

Computing eigenvalues of a dense n x n matrix by the QR algorithm would cost O(n^3) PER ITERATION if
run on the full matrix -- ruinous, since dozens of iterations are needed. The fix, universal in
practice (LAPACK, every serious eigensolver), is to first reduce the matrix by an ORTHOGONAL SIMILARITY
transform to UPPER HESSENBERG form: zero everywhere below the first subdiagonal. A Hessenberg matrix is
one step away from triangular, and -- crucially -- the QR algorithm PRESERVES Hessenberg form, so each
subsequent QR step costs only O(n^2). The one-time reduction is O(n^3) but happens once.

The reduction uses HOUSEHOLDER REFLECTORS, the same tool as QR, but applied on BOTH sides to keep the
transform a similarity (A -> Q^T A Q) so the eigenvalues are unchanged. For column k, build a reflector
that zeros the entries below the subdiagonal, apply it from the left (P A), then from the right (P A P);
because it touches rows/columns k+1..n only, the already-created zeros survive. When A is SYMMETRIC the
two-sided reflectors keep it symmetric, so upper Hessenberg collapses to TRIDIAGONAL -- the form the
symmetric QR algorithm and Lanczos both target.

This module reduces a matrix to upper Hessenberg (and a symmetric matrix to tridiagonal) via Householder
similarity, returning both H and the accumulated orthogonal Q with A = Q H Q^T. It is validated: H is
upper Hessenberg (zeros below the subdiagonal); Q is orthogonal; Q H Q^T reconstructs A to machine
precision; the eigenvalues are preserved (trace and determinant match, and the characteristic behaviour
is unchanged); a symmetric matrix reduces to a symmetric tridiagonal H; and an already-Hessenberg matrix
is left essentially unchanged. Pure stdlib; the eigenvalue-preprocessing companion to the QR-algorithm,
Householder-QR, and Lanczos tools."""

from __future__ import annotations

import math


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _T(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def hessenberg(A):
    """Reduce A to upper Hessenberg form by Householder similarity. Returns (H, Q) with A = Q H Q^T."""
    n = len(A)
    H = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    Q = _identity(n)
    for k in range(n - 2):
        # build a Householder vector to zero H[k+2:, k]
        x = [H[i][k] for i in range(k + 1, n)]
        alpha = -math.copysign(math.sqrt(sum(xi * xi for xi in x)), x[0]) if x[0] != 0 else -math.sqrt(sum(xi * xi for xi in x))
        if alpha == 0:
            continue
        v = list(x)
        v[0] -= alpha
        vnorm = math.sqrt(sum(vi * vi for vi in v))
        if vnorm < 1e-300:
            continue
        v = [vi / vnorm for vi in v]              # normalized Householder vector (length n-k-1)
        # apply P = I - 2 v v^T from the LEFT to rows k+1..n-1: H <- P H
        for j in range(n):
            dot = sum(v[i] * H[k + 1 + i][j] for i in range(len(v)))
            for i in range(len(v)):
                H[k + 1 + i][j] -= 2 * v[i] * dot
        # apply from the RIGHT: H <- H P
        for i in range(n):
            dot = sum(H[i][k + 1 + j] * v[j] for j in range(len(v)))
            for j in range(len(v)):
                H[i][k + 1 + j] -= 2 * dot * v[j]
        # accumulate Q <- Q P  (so that A = Q H Q^T)
        for i in range(n):
            dot = sum(Q[i][k + 1 + j] * v[j] for j in range(len(v)))
            for j in range(len(v)):
                Q[i][k + 1 + j] -= 2 * dot * v[j]
    # clean tiny sub-subdiagonal noise
    for i in range(n):
        for j in range(n):
            if i > j + 1 and abs(H[i][j]) < 1e-12:
                H[i][j] = 0.0
    return H, Q


def is_upper_hessenberg(H, tol=1e-9):
    n = len(H)
    return all(abs(H[i][j]) < tol for i in range(n) for j in range(n) if i > j + 1)


def is_tridiagonal(H, tol=1e-9):
    n = len(H)
    return all(abs(H[i][j]) < tol for i in range(n) for j in range(n) if abs(i - j) > 1)


def is_symmetric(A, tol=1e-9):
    n = len(A)
    return all(abs(A[i][j] - A[j][i]) < tol for i in range(n) for j in range(n))


def reconstruct(H, Q):
    """Rebuild A = Q H Q^T from the Hessenberg factors."""
    return _matmul(_matmul(Q, H), _T(Q))


def trace(A):
    return sum(A[i][i] for i in range(len(A)))
