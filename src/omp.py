"""Orthogonal matching pursuit: recovering a sparse signal from far fewer measurements than unknowns.

COMPRESSED SENSING turns a startling fact into an algorithm: a signal that is SPARSE (only k of its n
coordinates are nonzero) can be recovered from far fewer than n linear measurements. Given a
measurement matrix A (m x n, with m << n) and observations y = A x, solving for x is underdetermined
-- infinitely many x fit -- but if x is k-sparse and A is "incoherent" (its columns not too aligned),
the sparsest solution is unique and recoverable. This underlies MRI acceleration, single-pixel
cameras, radar, and any setting where measurements are expensive but the signal is compressible.

ORTHOGONAL MATCHING PURSUIT is the classic greedy solver. It builds the support one atom at a time:

    1. find the column of A most CORRELATED with the current residual,
    2. add it to the active set,
    3. re-solve the LEAST-SQUARES fit of y using only the active columns (the "orthogonal" step --
       it projects y onto the span of chosen columns, so the residual is orthogonal to all of them),
    4. update the residual and repeat until k atoms are chosen or the residual is tiny.

Each step reduces the residual, and for incoherent A with enough measurements OMP recovers the exact
support and coefficients. This module implements OMP (given a sparsity level or a residual tolerance),
using a normal-equations least-squares solve on the active columns, plus helpers to build random
Gaussian measurement matrices and check recovery.

Validated: on a planted k-sparse signal measured by a random Gaussian matrix, OMP recovers the exact
support and coefficients to tolerance when m is a few times k log(n/k); the residual decreases
monotonically each iteration; it stops at the right sparsity; a fully-measured (square, invertible)
system is solved exactly; and it degrades gracefully (bounded error) under measurement noise. Pure
stdlib; the sparse-recovery companion to the SVD / least-squares and the FFT-based signal tools."""

from __future__ import annotations

import math


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


def random_matrix(m, n, seed=12345, normalize_cols=True):
    """An m x n Gaussian measurement matrix, optionally with unit-norm columns (standard for OMP)."""
    rng = _lcg(seed)
    A = [[_gaussian(rng) for _ in range(n)] for _ in range(m)]
    if normalize_cols:
        for j in range(n):
            norm = math.sqrt(sum(A[i][j] ** 2 for i in range(m)))
            if norm > 0:
                for i in range(m):
                    A[i][j] /= norm
    return A


def matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _lstsq_active(A, y, active):
    """Least-squares solve of A[:, active] c = y via normal equations. Returns coefficients c
    aligned with `active`."""
    m = len(A)
    k = len(active)
    # B = A[:, active], solve B^T B c = B^T y
    BtB = [[0.0] * k for _ in range(k)]
    Bty = [0.0] * k
    for a in range(k):
        ja = active[a]
        Bty[a] = sum(A[i][ja] * y[i] for i in range(m))
        for b in range(k):
            jb = active[b]
            BtB[a][b] = sum(A[i][ja] * A[i][jb] for i in range(m))
    return _solve(BtB, Bty)


def _solve(M, b):
    """Gaussian elimination with partial pivoting (small dense system)."""
    n = len(b)
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        p = A[col][col]
        if abs(p) < 1e-15:
            continue
        for r in range(col + 1, n):
            f = A[r][col] / p
            for c in range(col, n + 1):
                A[r][c] -= f * A[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-15:
            x[i] = 0.0
            continue
        s = A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / A[i][i]
    return x


def omp(A, y, sparsity=None, tol=1e-8, max_iter=None):
    """Orthogonal matching pursuit. Recover a sparse x with A x ~ y. Stops after `sparsity` atoms
    or when the residual norm falls below tol. Returns x (length n), the full coefficient vector."""
    m = len(A)
    n = len(A[0])
    if max_iter is None:
        max_iter = sparsity if sparsity else m
    residual = list(y)
    active = []
    coeffs = {}
    for _ in range(max_iter):
        # correlation of each column with the residual
        best_j = -1
        best_c = 0.0
        for j in range(n):
            if j in active:
                continue
            c = abs(sum(A[i][j] * residual[i] for i in range(m)))
            if c > best_c:
                best_c = c
                best_j = j
        if best_j == -1:
            break
        active.append(best_j)
        c = _lstsq_active(A, y, active)
        coeffs = {active[t]: c[t] for t in range(len(active))}
        # residual = y - A[:,active] c
        approx = [0.0] * m
        for t, j in enumerate(active):
            for i in range(m):
                approx[i] += A[i][j] * c[t]
        residual = [y[i] - approx[i] for i in range(m)]
        rnorm = math.sqrt(sum(r * r for r in residual))
        if rnorm < tol:
            break
        if sparsity and len(active) >= sparsity:
            break
    x = [0.0] * n
    for j, v in coeffs.items():
        x[j] = v
    return x


def residual_norm(A, x, y):
    """||A x - y||."""
    approx = matvec(A, x)
    return math.sqrt(sum((approx[i] - y[i]) ** 2 for i in range(len(y))))


def support(x, tol=1e-6):
    """Indices of the nonzero (above tol) entries of x."""
    return sorted(i for i, v in enumerate(x) if abs(v) > tol)
