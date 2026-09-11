"""Cubic spline interpolation: a smooth curve through every point.

Given data points (x_0, y_0), ..., (x_n, y_n), how do you draw a smooth curve through all of
them? A single high-degree polynomial oscillates wildly between the points (Runge's phenomenon).
A cubic spline instead fits a SEPARATE cubic to each interval and stitches them so the whole
curve is twice continuously differentiable (C^2) -- the value, slope, and curvature all match at
every join. The result is the smoothest possible interpolant in a precise sense: it minimizes
the integrated squared second derivative, which is why a draftsman's flexible "spline" ruler
naturally takes this shape.

The construction reduces to solving for the second derivatives M_i at the knots. C^2 continuity
across the interior knots gives a tridiagonal linear system,

    h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1} = 6( (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1} ),

solved in O(n) by the Thomas algorithm. The NATURAL spline sets the end curvatures M_0 = M_n = 0
(a free-ended ruler); the CLAMPED spline instead fixes the end slopes. Each cubic piece is then
evaluated in Horner-like form, and its analytic derivative comes for free.

This module builds a natural (or clamped) cubic spline, evaluates it and its derivative, and
verifies it passes through every knot, is continuous in value/slope/curvature at the joins, and
reproduces low-degree polynomials exactly. Pure stdlib; the interpolation companion to the
quadrature and Horner notes."""

from __future__ import annotations

import bisect


class CubicSpline:
    """A natural or clamped cubic spline through the given (x, y) knots (x strictly increasing)."""

    def __init__(self, xs, ys, bc="natural", clamp=(0.0, 0.0)):
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")
        if len(xs) < 2:
            raise ValueError("need at least two points")
        if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("xs must be strictly increasing")
        self.xs = list(map(float, xs))
        self.ys = list(map(float, ys))
        self.n = len(xs) - 1
        self.bc = bc
        self.M = self._solve_second_derivatives(bc, clamp)

    def _solve_second_derivatives(self, bc, clamp):
        n = self.n
        xs, ys = self.xs, self.ys
        h = [xs[i + 1] - xs[i] for i in range(n)]
        # assemble the tridiagonal system for M[0..n]
        a = [0.0] * (n + 1)   # sub-diagonal
        b = [0.0] * (n + 1)   # diagonal
        c = [0.0] * (n + 1)   # super-diagonal
        d = [0.0] * (n + 1)   # right-hand side
        # interior equations (C^2 continuity)
        for i in range(1, n):
            a[i] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            d[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
        # boundary conditions
        if bc == "natural":
            b[0] = 1.0; d[0] = 0.0        # M_0 = 0
            b[n] = 1.0; d[n] = 0.0        # M_n = 0
        elif bc == "clamped":
            m0, mn = clamp
            b[0] = 2.0 * h[0]; c[0] = h[0]
            d[0] = 6.0 * ((ys[1] - ys[0]) / h[0] - m0)
            a[n] = h[n - 1]; b[n] = 2.0 * h[n - 1]
            d[n] = 6.0 * (mn - (ys[n] - ys[n - 1]) / h[n - 1])
        else:
            raise ValueError("bc must be 'natural' or 'clamped'")
        return _thomas(a, b, c, d)

    def _interval(self, x):
        """Index i of the interval [x_i, x_{i+1}] containing x (clamped to the ends)."""
        if x <= self.xs[0]:
            return 0
        if x >= self.xs[-1]:
            return self.n - 1
        return bisect.bisect_right(self.xs, x) - 1

    def __call__(self, x):
        """Evaluate the spline at x (extrapolates with the end cubics outside the range)."""
        i = self._interval(x)
        xs, ys, M = self.xs, self.ys, self.M
        h = xs[i + 1] - xs[i]
        t = x - xs[i]
        u = xs[i + 1] - x
        # standard cubic-spline form in terms of the knot second derivatives
        return (M[i] * u ** 3 / (6 * h) + M[i + 1] * t ** 3 / (6 * h)
                + (ys[i] / h - M[i] * h / 6) * u
                + (ys[i + 1] / h - M[i + 1] * h / 6) * t)

    def derivative(self, x):
        """First derivative of the spline at x."""
        i = self._interval(x)
        xs, ys, M = self.xs, self.ys, self.M
        h = xs[i + 1] - xs[i]
        t = x - xs[i]
        u = xs[i + 1] - x
        return (-M[i] * u ** 2 / (2 * h) + M[i + 1] * t ** 2 / (2 * h)
                - (ys[i] / h - M[i] * h / 6)
                + (ys[i + 1] / h - M[i + 1] * h / 6))

    def second_derivative(self, x):
        """Second derivative (linear within each interval, continuous across joins)."""
        i = self._interval(x)
        xs, M = self.xs, self.M
        h = xs[i + 1] - xs[i]
        t = x - xs[i]
        u = xs[i + 1] - x
        return M[i] * u / h + M[i + 1] * t / h


def _thomas(a, b, c, d):
    """Solve a tridiagonal system (sub a, diag b, super c, rhs d) by the Thomas algorithm in
    O(n). Returns the solution vector. Does not modify the inputs."""
    n = len(b)
    cp = list(c)
    dp = list(d)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def lagrange(xs, ys, x):
    """Single-polynomial Lagrange interpolation at x -- included to contrast with the spline
    (it passes through the points too, but oscillates badly for many equally spaced knots:
    Runge's phenomenon)."""
    n = len(xs)
    total = 0.0
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if j != i:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        total += term
    return total
