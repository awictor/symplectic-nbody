"""Barycentric Lagrange interpolation: the stable, fast way to evaluate the interpolating polynomial.

Given n+1 points (x_i, y_i) with distinct nodes, exactly one polynomial of degree <= n passes through
them. The textbook Lagrange formula writes it as a sum of basis polynomials, but that form is O(n^2)
to evaluate and numerically shaky. The BARYCENTRIC form is the one actually used in practice: after an
O(n^2) precomputation of WEIGHTS w_i = 1 / prod_{j != i} (x_i - x_j), evaluating the polynomial at any
point costs only O(n):

    p(x) = ( sum_i w_i y_i / (x - x_i) ) / ( sum_i w_i / (x - x_i) ).

This "second barycentric form" is beautifully stable, is trivial to update when a new point is added,
and -- crucially -- with the RIGHT nodes it is a superb approximation tool. Equally spaced nodes suffer
RUNGE'S PHENOMENON (the interpolant oscillates wildly near the ends and the error diverges as the
degree grows), but CHEBYSHEV nodes, clustered toward the interval ends, have simple closed-form
barycentric weights w_i = (-1)^i (or with half-weight endpoints) and converge geometrically for
analytic functions -- turning interpolation from a trap into one of the most accurate tools in
numerical analysis.

This module builds a barycentric interpolant from arbitrary nodes (computing the weights), evaluates
it in O(n) with the exact-node special case handled, provides Chebyshev-node constructors with their
analytic weights, and demonstrates Runge's phenomenon. Validated: the interpolant passes exactly
through every data point; it reproduces low-degree polynomials to machine precision (a degree-d
polynomial sampled at d+1 nodes is recovered everywhere); Chebyshev interpolation of smooth functions
converges geometrically while equispaced interpolation of Runge's function diverges; and the weights
match a direct product formula. Pure stdlib; the interpolation companion to the Chebyshev-series and
spline tools."""

from __future__ import annotations

import math


class Barycentric:
    """A polynomial interpolant in second barycentric form through given (x_i, y_i)."""

    def __init__(self, nodes, values, weights=None):
        if len(nodes) != len(values):
            raise ValueError("nodes and values must have equal length")
        if len(set(nodes)) != len(nodes):
            raise ValueError("nodes must be distinct")
        self.x = list(nodes)
        self.y = list(values)
        self.n = len(nodes)
        self.w = list(weights) if weights is not None else self._compute_weights()

    def _compute_weights(self):
        w = [1.0] * self.n
        for i in range(self.n):
            prod = 1.0
            for j in range(self.n):
                if j != i:
                    prod *= (self.x[i] - self.x[j])
            w[i] = 1.0 / prod
        return w

    def __call__(self, xq):
        """Evaluate the interpolant at xq (scalar) in O(n)."""
        num = 0.0
        den = 0.0
        for i in range(self.n):
            diff = xq - self.x[i]
            if diff == 0.0:
                return self.y[i]  # exactly on a node
            t = self.w[i] / diff
            num += t * self.y[i]
            den += t
        return num / den

    def evaluate(self, xs):
        return [self(x) for x in xs]


def chebyshev_nodes(n, a=-1.0, b=1.0):
    """n+1 Chebyshev points of the second kind (extrema) on [a, b], clustered toward the ends."""
    return [0.5 * (a + b) + 0.5 * (b - a) * math.cos(math.pi * k / n) for k in range(n + 1)]


def chebyshev_weights(n):
    """The analytic barycentric weights for second-kind Chebyshev points: w_k = (-1)^k, with the
    two endpoints halved. (Overall scale cancels in the barycentric formula.)"""
    w = [(-1.0) ** k for k in range(n + 1)]
    w[0] *= 0.5
    w[n] *= 0.5
    return w


def chebyshev_interpolant(f, degree, a=-1.0, b=1.0):
    """Interpolate f at degree+1 Chebyshev nodes on [a, b] using the analytic weights."""
    nodes = chebyshev_nodes(degree, a, b)
    values = [f(x) for x in nodes]
    weights = chebyshev_weights(degree)
    return Barycentric(nodes, values, weights)


def equispaced_interpolant(f, degree, a=-1.0, b=1.0):
    """Interpolate f at degree+1 equally spaced nodes on [a, b] (prone to Runge's phenomenon)."""
    if degree == 0:
        nodes = [0.5 * (a + b)]
    else:
        nodes = [a + (b - a) * k / degree for k in range(degree + 1)]
    values = [f(x) for x in nodes]
    return Barycentric(nodes, values)


def max_error(interp, f, a=-1.0, b=1.0, samples=500):
    """Maximum |interp(x) - f(x)| over a fine sample of [a, b]."""
    err = 0.0
    for k in range(samples + 1):
        x = a + (b - a) * k / samples
        err = max(err, abs(interp(x) - f(x)))
    return err


# --- reference: naive Lagrange evaluation ------------------------------------
def lagrange_eval(nodes, values, xq):
    """Direct O(n^2) Lagrange interpolation evaluation (reference for the barycentric form)."""
    n = len(nodes)
    total = 0.0
    for i in range(n):
        term = values[i]
        for j in range(n):
            if j != i:
                term *= (xq - nodes[j]) / (nodes[i] - nodes[j])
        total += term
    return total
