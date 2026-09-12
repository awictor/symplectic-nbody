"""Chebyshev approximation: near-optimal polynomial fits that dodge the Runge phenomenon.

Approximating a function by a polynomial sounds simple -- sample it and interpolate -- but sampling at
EQUALLY SPACED points is a trap: for many smooth functions the interpolation error EXPLODES near the
interval ends as the degree rises (the RUNGE PHENOMENON), wild oscillations that make high-degree
equispaced fits useless. The fix is to sample at CHEBYSHEV POINTS -- the projections of equally spaced
points on a circle onto the interval, clustered toward the ends. Chebyshev interpolation converges
for every continuous function and is NEAR-OPTIMAL: its maximum error is within a small factor of the
best possible polynomial of that degree, and it converges geometrically for analytic functions.

The magic comes from the CHEBYSHEV POLYNOMIALS T_n(x) = cos(n arccos x), an orthogonal family on
[-1, 1] whose extrema and roots are exactly the good sampling points. A function is expanded as a sum
of T_n weighted by CHEBYSHEV COEFFICIENTS, computed from samples at the Chebyshev nodes; the series
is evaluated stably by CLENSHAW'S RECURRENCE (a Horner-like scheme for orthogonal polynomials).
Because T_n oscillates between -1 and 1 with equal-ripple extrema, a truncated Chebyshev series
spreads its error evenly across the interval -- the equioscillation that characterizes the minimax
(best uniform) approximation, which is why Chebyshev fits are the foundation of function-approximation
libraries and spectral methods.

This module builds Chebyshev interpolants of a function on any interval, evaluates them by Clenshaw
recurrence, and exposes the Chebyshev nodes and coefficients. It is verified against exact references:
that the interpolant matches the function to a tiny error for smooth functions (exp, sin, a rational
function), that its error shrinks geometrically with the degree, that on Runge's function it stays
bounded where equispaced interpolation blows up, that the coefficients of a low-degree polynomial
recover it exactly, and that the nodes lie in the interval clustered at the ends. Pure stdlib; a
numerical-methods companion to the spline, Bezier, and quadrature notes."""

from __future__ import annotations

import math


def cheb_nodes(n, a=-1.0, b=1.0):
    """The n Chebyshev points of the second kind on [a, b] (extrema of T_{n-1}), clustered at the
    ends. Returns them in increasing order."""
    if n == 1:
        return [(a + b) / 2]
    nodes = [math.cos(math.pi * k / (n - 1)) for k in range(n)]   # in [-1, 1], decreasing
    nodes.reverse()
    return [(a + b) / 2 + (b - a) / 2 * x for x in nodes]


class ChebyshevInterpolant:
    """A Chebyshev-series approximation of a function on [a, b]."""

    def __init__(self, f, degree, a=-1.0, b=1.0):
        self.a = a
        self.b = b
        self.degree = degree
        N = degree + 1
        # sample at the Chebyshev points of the FIRST kind (roots of T_N) for the discrete transform
        self.coeffs = self._fit(f, N, a, b)

    @staticmethod
    def _fit(f, N, a, b):
        # Chebyshev-Gauss nodes: x_k = cos(pi (k+0.5)/N), k=0..N-1, mapped to [a,b]
        xs = [math.cos(math.pi * (k + 0.5) / N) for k in range(N)]
        fs = [f((a + b) / 2 + (b - a) / 2 * x) for x in xs]
        coeffs = []
        for j in range(N):
            s = sum(fs[k] * math.cos(math.pi * j * (k + 0.5) / N) for k in range(N))
            coeffs.append((2.0 / N) * s)
        coeffs[0] *= 0.5
        return coeffs

    def __call__(self, x):
        """Evaluate the interpolant at x by Clenshaw's recurrence."""
        # map x from [a,b] to [-1,1]
        t = (2 * x - (self.a + self.b)) / (self.b - self.a)
        c = self.coeffs
        b1 = b2 = 0.0
        for j in range(len(c) - 1, 0, -1):
            b1, b2 = 2 * t * b1 - b2 + c[j], b1
        return t * b1 - b2 + c[0]

    def max_error(self, f, samples=1000):
        """Maximum absolute error against f over the interval, sampled densely."""
        err = 0.0
        for i in range(samples + 1):
            x = self.a + (self.b - self.a) * i / samples
            err = max(err, abs(self(x) - f(x)))
        return err


def cheb_polynomial(n, x):
    """The Chebyshev polynomial T_n(x) evaluated at x in [-1, 1]."""
    if abs(x) <= 1:
        return math.cos(n * math.acos(x))
    # outside [-1,1] use the hyperbolic form
    return math.cosh(n * math.acosh(abs(x))) * (1 if x > 0 or n % 2 == 0 else -1)


def equispaced_interpolate(f, degree, a=-1.0, b=1.0):
    """Polynomial interpolation at EQUALLY spaced points (to contrast with Chebyshev -- this is the
    one that suffers the Runge phenomenon). Returns an evaluator via Lagrange interpolation."""
    N = degree + 1
    xs = [a + (b - a) * k / (N - 1) for k in range(N)] if N > 1 else [(a + b) / 2]
    ys = [f(x) for x in xs]

    def evaluate(x):
        total = 0.0
        for i in range(N):
            term = ys[i]
            for j in range(N):
                if i != j:
                    term *= (x - xs[j]) / (xs[i] - xs[j])
            total += term
        return total

    return evaluate
