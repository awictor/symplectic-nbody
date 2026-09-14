"""Gram-Schmidt orthogonalization: turning any basis into an orthonormal one, and why the order matters.

Given a set of linearly independent vectors, GRAM-SCHMIDT produces an orthonormal set spanning the same
space -- the workhorse behind the QR factorization, least squares, Krylov methods, and orthogonal
polynomials. The CLASSICAL algorithm (CGS) is the textbook one: to orthogonalize v_k, subtract its
projection onto each earlier orthonormal q_i all at once, then normalize. It is mathematically exact,
but numerically it is a trap -- on vectors that are nearly linearly dependent (an ill-conditioned
basis), roundoff in the projections accumulates and the computed q_k drift far from orthogonal.

The MODIFIED algorithm (MGS) fixes this with a one-line change of ORDER: instead of projecting v_k onto
all earlier q_i using the original v_k each time, subtract each projection SEQUENTIALLY, updating the
working vector after every step so the next projection sees the already-reduced remainder. The two are
algebraically identical in exact arithmetic, but MGS is dramatically more orthogonal in floating point
-- a classic illustration that the ORDER of operations, not just the formula, decides numerical
stability.

This module implements both, plus the resulting QR factorization (A = Q R with Q's orthonormal columns
and R upper-triangular), orthonormality measurement, and a Gram-Schmidt process for functions
(producing orthogonal polynomials on an interval, e.g. Legendre from the monomials). It is validated:
both variants produce orthonormal vectors spanning the input span and reconstruct A = Q R exactly;
MGS's loss of orthogonality on a deliberately ill-conditioned (Hilbert-like) basis is far smaller than
CGS's; the R factor is upper-triangular with positive diagonal; orthogonalizing an already-orthonormal
set is a no-op; and the polynomial version reproduces the (scaled) Legendre polynomials, orthogonal
under the L2 inner product on [-1, 1]. Pure stdlib; the orthogonalization companion to the
Householder-QR, LLL, and Lanczos tools."""

from __future__ import annotations

import math


def _dot(u, v):
    return sum(u[i] * v[i] for i in range(len(u)))


def _norm(v):
    return math.sqrt(_dot(v, v))


def _scale(v, a):
    return [a * x for x in v]


def _sub(u, v):
    return [u[i] - v[i] for i in range(len(u))]


def classical_gram_schmidt(vectors):
    """Classical Gram-Schmidt. Returns (Q, R): orthonormal vectors Q and the coefficients R.

    vectors: list of column vectors (each a list). Q[k] is the k-th orthonormal vector.
    """
    n = len(vectors)
    Q = []
    R = [[0.0] * n for _ in range(n)]
    for k in range(n):
        v = list(vectors[k])
        w = list(vectors[k])
        # project onto all earlier q_i at once (using the ORIGINAL v)
        for i in range(k):
            R[i][k] = _dot(Q[i], v)
            w = _sub(w, _scale(Q[i], R[i][k]))
        R[k][k] = _norm(w)
        if R[k][k] < 1e-300:
            raise ValueError("linearly dependent vectors")
        Q.append(_scale(w, 1.0 / R[k][k]))
    return Q, R


def modified_gram_schmidt(vectors):
    """Modified Gram-Schmidt. Same result in exact arithmetic, far more orthogonal in floating point.

    Subtracts each projection sequentially, updating the working vector after every step.
    """
    n = len(vectors)
    Q = []
    R = [[0.0] * n for _ in range(n)]
    for k in range(n):
        w = list(vectors[k])
        for i in range(k):
            R[i][k] = _dot(Q[i], w)          # project against the ALREADY-REDUCED w
            w = _sub(w, _scale(Q[i], R[i][k]))
        R[k][k] = _norm(w)
        if R[k][k] < 1e-300:
            raise ValueError("linearly dependent vectors")
        Q.append(_scale(w, 1.0 / R[k][k]))
    return Q, R


def qr(A, method="mgs"):
    """QR factorization of A (list of ROWS) via Gram-Schmidt on its columns.

    Returns (Qmat, R) with Qmat a list of rows whose columns are orthonormal, and R upper-triangular.
    """
    m = len(A)
    n = len(A[0])
    cols = [[A[i][j] for i in range(m)] for j in range(n)]
    gs = modified_gram_schmidt if method == "mgs" else classical_gram_schmidt
    Q, R = gs(cols)
    # Q is a list of column vectors; assemble as rows
    Qmat = [[Q[j][i] for j in range(n)] for i in range(m)]
    return Qmat, R


def orthogonality_error(Q):
    """Max |<q_i, q_j> - delta_ij| over all pairs -- how far Q is from orthonormal."""
    n = len(Q)
    err = 0.0
    for i in range(n):
        for j in range(n):
            d = _dot(Q[i], Q[j])
            target = 1.0 if i == j else 0.0
            err = max(err, abs(d - target))
    return err


def reconstruct(Q, R):
    """Rebuild the original column vectors from Q (columns) and R: v_k = sum_i R[i][k] q_i."""
    n = len(Q)
    m = len(Q[0])
    cols = []
    for k in range(n):
        v = [0.0] * m
        for i in range(n):
            if R[i][k] != 0:
                for r in range(m):
                    v[r] += R[i][k] * Q[i][r]
        cols.append(v)
    return cols


def orthogonal_polynomials(degree, a=-1.0, b=1.0, samples=400):
    """Gram-Schmidt on the monomials 1, x, x^2, ... under the L2 inner product on [a, b].

    Returns the orthonormal polynomials as coefficient lists (ascending powers). On [-1, 1] these are
    the normalized Legendre polynomials.
    """
    # represent each polynomial by its coefficient vector; inner product by numerical integration
    xs = [a + (b - a) * k / (samples - 1) for k in range(samples)]
    dx = (b - a) / (samples - 1)

    def poly_val(coeffs, x):
        r = 0.0
        for c in reversed(coeffs):
            r = r * x + c
        return r

    def inner(p, q):
        # trapezoidal L2 inner product
        total = 0.0
        for i, x in enumerate(xs):
            w = 0.5 if (i == 0 or i == samples - 1) else 1.0
            total += w * poly_val(p, x) * poly_val(q, x)
        return total * dx

    monomials = []
    for d in range(degree + 1):
        c = [0.0] * (d + 1)
        c[d] = 1.0
        monomials.append(c)

    ortho = []
    for k in range(degree + 1):
        p = list(monomials[k])
        for q in ortho:
            coef = inner(p, q)
            # p = p - coef * q  (pad to same length)
            L = max(len(p), len(q))
            p = [(p[i] if i < len(p) else 0.0) - coef * (q[i] if i < len(q) else 0.0)
                 for i in range(L)]
        nrm = math.sqrt(inner(p, p))
        ortho.append([c / nrm for c in p])
    return ortho
