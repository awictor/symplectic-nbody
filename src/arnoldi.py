"""Arnoldi iteration: extract a few eigenvalues of a large, general (non-symmetric) matrix from a small subspace.

The full eigenvalue problem for an n-by-n matrix costs O(n^3) and stores the whole matrix; but you rarely
need all n eigenvalues -- usually just the few of largest magnitude (the dominant modes of a dynamical
system, the slowest-decaying states of a Markov chain, the leading resonances of a network). The ARNOLDI
ITERATION (1951) delivers exactly those from only matrix-vector products. Starting from one vector it
builds an orthonormal Krylov basis {q, Aq, A^2 q, ...} via modified Gram-Schmidt, and the projection of A
onto that basis is a small (m+1)-by-m upper HESSENBERG matrix H. The eigenvalues of its leading m-by-m
block -- the RITZ VALUES -- converge to the extremal eigenvalues of A remarkably fast, typically nailing
the largest few in m << n steps.

Arnoldi is the non-symmetric cousin of Lanczos (which needs a symmetric A and collapses H to a
tridiagonal), and the engine underneath ARPACK / MATLAB's `eigs` / GMRES. Its defining identity is the
Arnoldi factorization

    A Q_m = Q_{m+1} H,    i.e.   A Q_m = Q_m H_m + h_{m+1,m} q_{m+1} e_m^T,

with Q_m^T Q_m = I. This module implements Arnoldi with modified Gram-Schmidt reorthogonalization, exposes
the factorization (Q, H), computes Ritz values (via the repo's QR-algorithm eigensolver, which handles the
complex eigenvalues a non-symmetric H can have) and Ritz vectors, and includes a happy-breakdown check
(an exact invariant subspace found early). It is validated: the Arnoldi identity A Q_m = Q_{m+1} H holds
to machine precision; Q has orthonormal columns; H is upper Hessenberg; the Ritz values match the true
dominant eigenvalues of a non-symmetric matrix (cross-checked against the QR-algorithm on the full matrix)
and of a symmetric one (against Lanczos); a rank-deficient Krylov space triggers a clean happy breakdown;
and results are reproducible per seed. Pure stdlib; the Krylov-subspace companion to the Lanczos, GMRES,
conjugate-gradient, and QR-algorithm tools."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import qr_algorithm  # full-matrix eigensolver, handles complex spectra


def _dot(u, v):
    return sum(u[i] * v[i] for i in range(len(u)))


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


class _Rng:
    """Seeded LCG for a reproducible starting vector."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def normal(self):
        # Box-Muller with a cached spare
        if getattr(self, "_spare", None) is not None:
            s = self._spare
            self._spare = None
            return s
        u1 = self._u()
        u2 = self._u()
        u1 = max(u1, 1e-12)
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)

    def _u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def matvec_from_matrix(A):
    """Return a matvec closure x -> A x for a dense matrix A."""
    n = len(A)

    def mv(x):
        return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    return mv


def arnoldi_factorization(matvec, n, m, v0=None, seed=12345, tol=1e-12):
    """Build the Arnoldi factorization A Q_m = Q_{m+1} H via modified Gram-Schmidt.

    Returns (Q, H, breakdown_at) where:
      Q is a list of m+1 orthonormal basis vectors (each length n),
      H is (k+1)-by-k upper Hessenberg (k = actual steps taken),
      breakdown_at is the step index of a happy breakdown, or None.
    Reorthogonalizes once (modified Gram-Schmidt twice) for numerical stability."""
    if v0 is None:
        rng = _Rng(seed)
        v0 = [rng.normal() for _ in range(n)]
    beta = _norm(v0)
    if beta < tol:
        raise ValueError("start vector is (near) zero")
    q = [x / beta for x in v0]
    Q = [q]
    # H stored as list of columns; we build it row-friendly as a dict then densify
    H_cols = []
    breakdown_at = None
    for j in range(m):
        w = matvec(Q[j])
        h_col = [0.0] * (j + 2)
        # modified Gram-Schmidt, twice (reorthogonalization)
        for _pass in range(2):
            for i in range(j + 1):
                proj = _dot(Q[i], w)
                h_col[i] += proj
                w = [w[k] - proj * Q[i][k] for k in range(n)]
        hnext = _norm(w)
        h_col[j + 1] = hnext
        H_cols.append(h_col)
        if hnext < tol:
            # happy breakdown: Krylov space is A-invariant, exact eigenpairs available
            breakdown_at = j + 1
            break
        Q.append([x / hnext for x in w])
    k = len(H_cols)
    # densify H to (k+1)-by-k
    H = [[0.0] * k for _ in range(k + 1)]
    for j in range(k):
        for i in range(len(H_cols[j])):
            H[i][j] = H_cols[j][i]
    return Q, H, breakdown_at


def _leading_block(H):
    """Return the leading k-by-k block of the (k+1)-by-k Hessenberg matrix."""
    k = len(H[0])
    return [[H[i][j] for j in range(k)] for i in range(k)]


def ritz_values(matvec, n, m, v0=None, seed=12345):
    """Approximate the dominant eigenvalues of A as the eigenvalues of the m-by-m Hessenberg block.

    Returns the Ritz values sorted by DESCENDING magnitude (dominant first). Complex where the
    non-symmetric H demands it."""
    Q, H, brk = arnoldi_factorization(matvec, n, m, v0=v0, seed=seed)
    Hk = _leading_block(H)
    eigs = qr_algorithm.eigenvalues(Hk)
    eigs.sort(key=lambda z: -abs(z))
    return eigs


def ritz_pairs(matvec, n, m, v0=None, seed=12345):
    """Return (ritz_value, ritz_vector) pairs, dominant first.

    A Ritz vector is Q_m y for an eigenvector y of the Hessenberg block; it approximates an
    eigenvector of A. Uses the power-basis eigenvector recovery for the real Ritz values (a
    residual-minimizing solve of (Hk - lambda I) y = 0 for each real eigenvalue)."""
    Q, H, brk = arnoldi_factorization(matvec, n, m, v0=v0, seed=seed)
    Hk = _leading_block(H)
    k = len(Hk)
    eigs = qr_algorithm.eigenvalues(Hk)
    pairs = []
    for lam in eigs:
        if abs(lam.imag) > 1e-9:
            continue  # skip genuinely complex Ritz values for real-vector recovery
        lr = lam.real
        y = _null_vector(Hk, lr)
        if y is None:
            continue
        # lift into R^n: x = Q_m y
        x = [sum(Q[j][i] * y[j] for j in range(k)) for i in range(n)]
        xn = _norm(x)
        if xn > 0:
            x = [xi / xn for xi in x]
        pairs.append((lr, x))
    pairs.sort(key=lambda p: -abs(p[0]))
    return pairs


def _null_vector(A, lam):
    """Approximate a unit null vector of (A - lam I) by inverse iteration with a shifted matrix."""
    n = len(A)
    M = [[A[i][j] - (lam if i == j else 0.0) for j in range(n)] for i in range(n)]
    # small ridge to keep the shifted system solvable, then a few inverse-iteration steps
    for i in range(n):
        M[i][i] += 1e-10
    rng = _Rng(999)
    x = [rng.normal() for _ in range(n)]
    xn = _norm(x)
    x = [xi / xn for xi in x]
    for _ in range(30):
        y = _solve(M, x)
        if y is None:
            return None
        yn = _norm(y)
        if yn < 1e-300:
            return None
        x = [yi / yn for yi in y]
    return x


def _solve(A, b):
    """Gaussian elimination with partial pivoting; returns None if singular."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / pv
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def factorization_residual(matvec, Q, H):
    """Max-abs entry of A Q_m - Q_{m+1} H -- should be ~machine epsilon. Measures Arnoldi accuracy."""
    k = len(H[0])
    n = len(Q[0])
    worst = 0.0
    for j in range(k):
        aq = matvec(Q[j])
        # (Q_{m+1} H) column j = sum_i H[i][j] * Q[i]
        qh = [0.0] * n
        for i in range(j + 2):
            hij = H[i][j]
            if hij != 0.0:
                for r in range(n):
                    qh[r] += hij * Q[i][r]
        for r in range(n):
            worst = max(worst, abs(aq[r] - qh[r]))
    return worst
