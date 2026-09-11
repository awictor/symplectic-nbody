"""Power iteration: finding eigenvalues without solving the characteristic polynomial.

An eigenvector of a matrix A is a direction A only stretches, not rotates: A v = lambda v, with
the scale factor lambda the eigenvalue. They govern the modes of a vibrating structure, the
stationary distribution of a Markov chain, the axes of a data cloud (PCA), and Google's original
PageRank. For anything beyond 2x2 the characteristic polynomial det(A - lambda I) = 0 is a bad
way to find them; the standard approach is iterative.

POWER ITERATION is the simplest: start with a random vector, repeatedly multiply by A and
normalize. Because each multiply amplifies the component along the largest-|eigenvalue| direction
most, the vector converges to the DOMINANT eigenvector, and the RAYLEIGH QUOTIENT v^T A v / v^T v
gives its eigenvalue. Convergence is geometric at the ratio |lambda_2 / lambda_1| of the two
largest eigenvalues.

Two companions extend it. INVERSE iteration applies power iteration to (A - mu I)^{-1}, whose
dominant eigenvalue corresponds to A's eigenvalue nearest the shift mu -- so it targets any
eigenvalue, and converges fast when mu is close. DEFLATION subtracts the found eigenpair
(lambda v v^T for a symmetric A) and re-runs power iteration to peel off the next one, recovering
the full spectrum of a symmetric matrix.

This module implements power iteration with the Rayleigh quotient, shifted inverse iteration, and
deflation, and checks the eigenvalues against A v = lambda v, the trace/determinant identities,
and small analytic cases. Pure stdlib; the linear-algebra companion to the QR and LU notes."""

from __future__ import annotations

import math

from linsolve import solve as _lu_solve      # reuse the LU solver for inverse iteration


def _matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _norm(v):
    return math.sqrt(_dot(v, v))


def _normalize(v):
    n = _norm(v)
    return [x / n for x in v] if n > 0 else v


def _default_start(n):
    """A generic non-degenerate start vector. Avoids the symmetric [1,1,...,1] guess, which can
    be exactly orthogonal to the wanted eigenvector (e.g. after deflation) and stall the
    iteration. Distinct, non-proportional entries have a component along every eigenvector."""
    return _normalize([math.sin(1.0 + i) + 1.3 for i in range(n)])


def rayleigh_quotient(A, v):
    """The Rayleigh quotient v^T A v / v^T v -- the best scalar estimate of the eigenvalue for a
    given approximate eigenvector v."""
    Av = _matvec(A, v)
    return _dot(v, Av) / _dot(v, v)


def power_iteration(A, x0=None, tol: float = 1e-12, max_iter: int = 2000):
    """Find the dominant eigenvalue (largest magnitude) and its eigenvector by power iteration.
    Returns (eigenvalue, eigenvector, iterations)."""
    n = len(A)
    v = _normalize(x0[:]) if x0 else _default_start(n)
    lam = rayleigh_quotient(A, v)
    for it in range(1, max_iter + 1):
        w = _matvec(A, v)
        nw = _norm(w)
        if nw < 1e-300:
            return 0.0, v, it        # A v = 0: v is in the null space
        v_new = [x / nw for x in w]
        lam_new = rayleigh_quotient(A, v_new)
        # align sign so the eigenvector doesn't flip each step (negative dominant eigenvalue)
        if _dot(v, v_new) < 0:
            v_new = [-x for x in v_new]
        if abs(lam_new - lam) <= tol * (1 + abs(lam_new)):
            return lam_new, v_new, it
        v, lam = v_new, lam_new
    return lam, v, max_iter


def inverse_iteration(A, shift: float, x0=None, tol: float = 1e-12, max_iter: int = 500):
    """Shifted inverse iteration: find the eigenvalue of A NEAREST `shift` and its eigenvector,
    by power-iterating (A - shift I)^{-1}. Returns (eigenvalue, eigenvector, iterations)."""
    n = len(A)
    B = [[A[i][j] - (shift if i == j else 0.0) for j in range(n)] for i in range(n)]
    v = _normalize(x0[:]) if x0 else _default_start(n)
    lam = rayleigh_quotient(A, v)
    for it in range(1, max_iter + 1):
        try:
            w = _lu_solve(B, v)          # solve (A - shift I) w = v  == multiply by the inverse
        except ValueError:
            # shift is (nearly) an exact eigenvalue -> v already close; return it
            return rayleigh_quotient(A, v), v, it
        v_new = _normalize(w)
        if _dot(v, v_new) < 0:
            v_new = [-x for x in v_new]
        lam_new = rayleigh_quotient(A, v_new)
        if abs(lam_new - lam) <= tol * (1 + abs(lam_new)):
            return lam_new, v_new, it
        v, lam = v_new, lam_new
    return lam, v, max_iter


def _deflate_symmetric(A, lam, v):
    """Return A - lam v v^T (v unit) -- removes the eigenpair (lam, v) from a symmetric A so the
    next power iteration finds the next-largest eigenvalue."""
    n = len(A)
    return [[A[i][j] - lam * v[i] * v[j] for j in range(n)] for i in range(n)]


def eigenvalues_symmetric(A, tol: float = 1e-11):
    """All eigenvalues of a SYMMETRIC matrix, largest-magnitude first, by power iteration with
    deflation. Returns (eigenvalues, eigenvectors)."""
    n = len(A)
    B = [row[:] for row in A]
    vals = []
    vecs = []
    for _ in range(n):
        lam, v, _ = power_iteration(B, tol=tol)
        vals.append(lam)
        vecs.append(v)
        B = _deflate_symmetric(B, lam, v)
    return vals, vecs


def characteristic_2x2(A):
    """Exact eigenvalues of a 2x2 matrix from the characteristic polynomial (for checking):
    lambda = (tr +- sqrt(tr^2 - 4 det)) / 2."""
    tr = A[0][0] + A[1][1]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    disc = tr * tr - 4 * det
    s = math.sqrt(abs(disc))
    if disc >= 0:
        return sorted([(tr + s) / 2, (tr - s) / 2], key=abs, reverse=True)
    return [complex(tr / 2, s / 2), complex(tr / 2, -s / 2)]      # complex pair


def trace(A):
    return sum(A[i][i] for i in range(len(A)))


def residual_norm(A, lam, v):
    """||A v - lambda v|| -- how well (lam, v) satisfies the eigenvalue equation."""
    Av = _matvec(A, v)
    return _norm([Av[i] - lam * v[i] for i in range(len(v))])
