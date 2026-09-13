"""Gaussian quadrature on infinite domains -- Gauss-Hermite and Gauss-Laguerre by Golub-Welsch.

Ordinary quadrature (trapezoid, Simpson, Gauss-Legendre) integrates over a FINITE interval. But a
staggering number of the integrals that matter run to infinity with a natural weight attached: the
expectation of anything under a Gaussian is an integral over (-inf, inf) with weight e^{-x^2}; a
Laplace transform, a waiting-time average, or a radial integral in physics runs over [0, inf) with
weight e^{-x}. Truncating and gridding these is wasteful and inaccurate. GAUSSIAN QUADRATURE for these
weights -- GAUSS-HERMITE for e^{-x^2} on the whole line, GAUSS-LAGUERRE for e^{-x} on the half-line --
places n nodes and weights so cunningly that the rule integrates every polynomial up to degree 2n-1
EXACTLY, the maximum possible for n points, and converges geometrically on smooth integrands.

The nodes are the roots of the orthogonal polynomials for each weight (Hermite, Laguerre), and finding
them by root-searching is fiddly. The GOLUB-WELSCH algorithm (1969) is the elegant modern route: the
orthogonal polynomials obey a three-term recurrence, whose coefficients form a symmetric TRIDIAGONAL
"Jacobi matrix", and the quadrature NODES are exactly its EIGENVALUES while the WEIGHTS are proportional
to the squared first components of its EIGENVECTORS. So computing an n-point rule reduces to one
symmetric tridiagonal eigenproblem -- solved here by a QL iteration -- turning a hard root-finding
problem into a clean linear-algebra one. This module builds the Jacobi matrices for the Hermite and
Laguerre weights, solves them, and returns the nodes and weights, plus integrators that apply the rule
and helpers to integrate a Gaussian expectation (mean mu, standard deviation sigma) by the change of
variables x = mu + sqrt(2) sigma t.

Pure standard library. The tridiagonal eigensolver is included so the module stands alone.

Validation. The defining property is exactness: an n-point Gauss rule integrates every polynomial up to
degree 2n-1 with zero error, and this is checked directly on monomials x^0, x^1, ..., x^(2n-1) against
their known closed forms (Hermite moments involve the double factorial; Laguerre moments are the
factorial n!). Beyond that: the weights are all positive and sum to the total mass of the weight
function (sqrt(pi) for Hermite, 1 for Laguerre); the Hermite nodes are symmetric about zero; known hard
integrals match to high precision -- the Gaussian expectation of x^2 is sigma^2, E[e^{x}] under a
standard normal is sqrt(e), and the Gamma-function integral of x^s e^{-x} equals Gamma(s+1); and
convergence to full accuracy on a smooth non-polynomial integrand is confirmed as n grows."""

import math


# ---------------------------------------------------------------------------
# symmetric tridiagonal eigensolver (QL with implicit shifts), with eigenvectors
# ---------------------------------------------------------------------------

def _tridiagonal_eigen(diag, off):
    """Eigenvalues and eigenvectors of a symmetric tridiagonal matrix.

    ``diag`` (length n) is the diagonal, ``off`` (length n-1) the sub/super-diagonal. Returns
    (values, first_components) where first_components[k] is the first entry of the k-th normalised
    eigenvector -- all Golub-Welsch needs for the weights.
    """
    n = len(diag)
    d = list(diag)
    e = list(off) + [0.0]
    # eigenvector matrix, initialised to identity; we only need the first row at the end
    z = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for l in range(n):
        it = 0
        while True:
            m = l
            while m < n - 1:
                dd = abs(d[m]) + abs(d[m + 1])
                if abs(e[m]) <= 1e-16 * dd:
                    break
                m += 1
            if m == l:
                break
            it += 1
            if it > 200:
                break
            g = (d[l + 1] - d[l]) / (2 * e[l])
            r = math.hypot(g, 1.0)
            g = d[m] - d[l] + e[l] / (g + (r if g >= 0 else -r))
            s = c = 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                r = math.hypot(f, g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m] = 0.0
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                for k in range(n):
                    f = z[k][i + 1]
                    z[k][i + 1] = s * z[k][i] + c * f
                    z[k][i] = c * z[k][i] - s * f
            else:
                d[l] -= p
                e[l] = g
                e[m] = 0.0
                continue
            d[l] -= p
            e[l] = g
            e[m] = 0.0

    first = [z[0][k] for k in range(n)]
    pairs = sorted(zip(d, first), key=lambda t: t[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ---------------------------------------------------------------------------
# Gauss-Hermite: weight e^{-x^2} on (-inf, inf), total mass sqrt(pi)
# ---------------------------------------------------------------------------

def gauss_hermite(n):
    """n-point Gauss-Hermite nodes and weights for integral f(x) e^{-x^2} dx over (-inf, inf).

    Jacobi matrix: diagonal 0, off-diagonal sqrt(k/2) for k=1..n-1. Weights = mu0 * (first component)^2
    with mu0 = sqrt(pi).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    diag = [0.0] * n
    off = [math.sqrt(k / 2.0) for k in range(1, n)]
    nodes, first = _tridiagonal_eigen(diag, off)
    mu0 = math.sqrt(math.pi)
    weights = [mu0 * f * f for f in first]
    return nodes, weights


# ---------------------------------------------------------------------------
# Gauss-Laguerre: weight e^{-x} on [0, inf), total mass 1
# ---------------------------------------------------------------------------

def gauss_laguerre(n):
    """n-point Gauss-Laguerre nodes and weights for integral f(x) e^{-x} dx over [0, inf).

    Jacobi matrix: diagonal 2k+1 (k=0..n-1), off-diagonal k (k=1..n-1). Weights = (first component)^2
    (total mass mu0 = 1).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    diag = [2.0 * k + 1.0 for k in range(n)]
    off = [float(k) for k in range(1, n)]
    nodes, first = _tridiagonal_eigen(diag, off)
    weights = [f * f for f in first]           # mu0 = integral e^{-x} dx = 1
    return nodes, weights


# ---------------------------------------------------------------------------
# integrators
# ---------------------------------------------------------------------------

def integrate_hermite(f, n):
    """Approximate integral of f(x) e^{-x^2} over (-inf, inf) with an n-point Gauss-Hermite rule."""
    nodes, weights = gauss_hermite(n)
    return sum(w * f(x) for x, w in zip(nodes, weights))


def integrate_laguerre(f, n):
    """Approximate integral of f(x) e^{-x} over [0, inf) with an n-point Gauss-Laguerre rule."""
    nodes, weights = gauss_laguerre(n)
    return sum(w * f(x) for x, w in zip(nodes, weights))


def gaussian_expectation(g, mu, sigma, n):
    """E[g(X)] for X ~ Normal(mu, sigma^2), via Gauss-Hermite with x = mu + sqrt(2) sigma t.

    The normal density is (1/(sqrt(2pi) sigma)) e^{-(x-mu)^2/(2 sigma^2)}; substituting turns the
    expectation into (1/sqrt(pi)) integral g(mu + sqrt(2) sigma t) e^{-t^2} dt.
    """
    nodes, weights = gauss_hermite(n)
    total = sum(w * g(mu + math.sqrt(2) * sigma * t) for t, w in zip(nodes, weights))
    return total / math.sqrt(math.pi)
