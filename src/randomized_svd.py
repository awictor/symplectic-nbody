"""Randomized SVD: near-optimal low-rank factorization by random projection, fast.

The singular value decomposition A = U S V^T is the gold standard for low-rank approximation, PCA,
and matrix compression -- but computing the full SVD of a large matrix costs O(m n min(m,n)), wasteful
when you only want the top k singular vectors. The RANDOMIZED SVD of Halko, Martinsson, and Tropp
(2011) gets those top k in roughly O(m n k) with a beautiful two-stage idea:

  STAGE A -- find a small subspace that captures A's range. Multiply A by a random Gaussian matrix
      Omega (n x (k+p), with p a few oversampling columns): Y = A Omega. Random directions, projected
      through A, almost surely span its dominant range. Orthonormalize Y (via QR / Gram-Schmidt) to
      get Q, an m x (k+p) basis with A ~ Q Q^T A.

  STAGE B -- do an exact SVD in the small space. Form B = Q^T A (only (k+p) x n), take its SVD
      B = U~ S V^T, and lift back: U = Q U~. Now A ~ U S V^T at rank k+p, truncated to k.

A couple of POWER ITERATIONS (replacing Y with (A A^T)^q A Omega) sharpen the approximation when the
singular values decay slowly, at the cost of a few more matrix multiplies. The remarkable guarantee is
that the expected error ||A - U S V^T|| is within a small factor of the best possible rank-k error
(the (k+1)-th singular value), with the gap shrinking as oversampling and power iterations grow.

This module implements randomized SVD with oversampling and optional power iteration, reusing the
repo's exact SVD for the small inner problem. Validated against that exact SVD: the recovered singular
values match the true top-k to tolerance, the low-rank reconstruction error is close to the optimal
(k+1)-th singular value, the singular vectors are orthonormal, power iteration reduces error on
slowly-decaying spectra, and an exactly rank-r matrix is recovered to machine precision. Pure stdlib;
the fast-approximate companion to the exact Jacobi SVD and the PCA / low-rank tools."""

from __future__ import annotations

import math

from svd import svd as exact_svd, _T, _matmul


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)

    return nxt


def _gaussian(rng):
    u1 = max(rng(), 1e-12)
    u2 = rng()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _random_matrix(rows, cols, rng):
    return [[_gaussian(rng) for _ in range(cols)] for _ in range(rows)]


def _qr_gram_schmidt(A):
    """Thin QR by modified Gram-Schmidt: return Q (same shape as A) with orthonormal columns."""
    m = len(A)
    n = len(A[0])
    # work on columns
    cols = [[A[i][j] for i in range(m)] for j in range(n)]
    Q = []
    for j in range(n):
        v = list(cols[j])
        for q in Q:
            dot = sum(q[i] * v[i] for i in range(m))
            v = [v[i] - dot * q[i] for i in range(m)]
        norm = math.sqrt(sum(x * x for x in v))
        if norm < 1e-12:
            continue  # dependent column, drop
        Q.append([x / norm for x in v])
    # return as an m x len(Q) matrix
    return [[Q[j][i] for j in range(len(Q))] for i in range(m)]


def randomized_svd(A, k, oversample=5, n_power=0, seed=12345):
    """Approximate the top-k SVD of A (m x n). Returns (U, S, Vt) with U m x k, S length-k,
    Vt k x n. oversample adds extra sampled columns; n_power power iterations sharpen accuracy."""
    m = len(A)
    n = len(A[0])
    ell = min(k + oversample, n, m)
    rng = _lcg(seed)

    # Stage A: random projection Y = A Omega
    Omega = _random_matrix(n, ell, rng)
    Y = _matmul(A, Omega)  # m x ell
    Q = _qr_gram_schmidt(Y)

    # power iterations: Q <- orth(A (A^T Q)) repeated
    At = _T(A)
    for _ in range(n_power):
        Z = _matmul(At, Q)          # n x ell
        Z = _qr_gram_schmidt(Z)
        Y = _matmul(A, Z)           # m x ell
        Q = _qr_gram_schmidt(Y)

    # Stage B: small SVD of B = Q^T A
    B = _matmul(_T(Q), A)           # ell x n
    Ub, S, Vt = exact_svd(B)
    # lift: U = Q Ub
    U = _matmul(Q, Ub)              # m x (rank of B)

    # truncate to k
    kk = min(k, len(S))
    U = [[U[i][j] for j in range(kk)] for i in range(len(U))]
    S = S[:kk]
    Vt = [Vt[j] for j in range(kk)]
    return U, S, Vt


def reconstruct(U, S, Vt):
    """Rebuild the rank-k approximation U diag(S) Vt."""
    m = len(U)
    k = len(S)
    n = len(Vt[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for t in range(k):
            us = U[i][t] * S[t]
            row = Vt[t]
            for j in range(n):
                out[i][j] += us * row[j]
    return out


def frobenius(A, B):
    """Frobenius norm of A - B."""
    s = 0.0
    for i in range(len(A)):
        for j in range(len(A[0])):
            d = A[i][j] - B[i][j]
            s += d * d
    return math.sqrt(s)
