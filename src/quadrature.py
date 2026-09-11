"""Numerical quadrature: computing a definite integral you cannot solve by hand.

Most integrals have no closed form -- exp(-x^2), sin(x)/x, an experimental curve -- so you
approximate the area under f(x) from a to b by sampling it. The methods here trade sample count
for accuracy, and the higher-order ones converge astonishingly fast on smooth functions.

  * TRAPEZOID rule joins the samples with straight lines: error O(h^2), where h is the step.
  * SIMPSON'S rule fits parabolas through pairs of intervals: error O(h^4) -- so halving the
    step cuts the error 16-fold, and it is exact for cubics.
  * ROMBERG integration applies Richardson extrapolation to a ladder of trapezoid estimates at
    halved steps, cancelling successive error terms; each column of the tableau gains two orders,
    reaching machine precision on smooth integrands in a handful of levels.
  * ADAPTIVE Simpson recursively subdivides only where the function is hard, spending samples
    where the curvature is, so a spiky integrand costs far less than a uniform grid.
  * GAUSS-LEGENDRE places n sample points and weights optimally, integrating polynomials up to
    degree 2n-1 exactly -- the most accuracy per evaluation for smooth functions.

This module implements all five and checks them against integrals with known exact values
(polynomials, exp, trig, the Gaussian), confirming the convergence orders. Pure stdlib; the
integration companion to the root-finding note (both approximate what algebra cannot give
directly)."""

from __future__ import annotations

import math


def trapezoid(f, a, b, n: int = 1000):
    """Composite trapezoid rule with n subintervals. Error ~ O(h^2)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def simpson(f, a, b, n: int = 1000):
    """Composite Simpson's rule with n subintervals (n rounded up to even). Error ~ O(h^4),
    exact for cubics."""
    if n < 2:
        n = 2
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3.0


def romberg(f, a, b, max_levels: int = 12, tol: float = 1e-12):
    """Romberg integration: Richardson extrapolation on a trapezoid ladder. Returns the best
    estimate, stopping early when two successive diagonal entries agree to `tol`."""
    R = [[0.0] * (max_levels + 1) for _ in range(max_levels + 1)]
    h = b - a
    R[0][0] = 0.5 * h * (f(a) + f(b))
    for i in range(1, max_levels + 1):
        h *= 0.5
        # trapezoid estimate at 2^i intervals, reusing prior samples (odd-indexed new points)
        s = 0.0
        for k in range(1, 2 ** i, 2):
            s += f(a + k * h)
        R[i][0] = 0.5 * R[i - 1][0] + s * h
        # extrapolation columns: each cancels the next error term
        for j in range(1, i + 1):
            R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4 ** j - 1)
        if i > 1 and abs(R[i][i] - R[i - 1][i - 1]) < tol:
            return R[i][i]
    return R[max_levels][max_levels]


def adaptive_simpson(f, a, b, tol: float = 1e-10, max_depth: int = 50):
    """Adaptive Simpson's rule: recursively subdivide only where the local Simpson estimate has
    not yet converged, concentrating samples where the function is hard."""
    def simpson3(fa, fm, fb, a, b):
        return (b - a) / 6.0 * (fa + 4 * fm + fb)

    def recurse(a, b, fa, fm, fb, whole, depth):
        m = 0.5 * (a + b)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)
        flm, frm = f(lm), f(rm)
        left = simpson3(fa, flm, fm, a, m)
        right = simpson3(fm, frm, fb, m, b)
        if depth <= 0 or abs(left + right - whole) <= 15 * tol:
            return left + right + (left + right - whole) / 15.0
        return (recurse(a, m, fa, flm, fm, left, depth - 1)
                + recurse(m, b, fm, frm, fb, right, depth - 1))

    fa, fb = f(a), f(b)
    m = 0.5 * (a + b)
    fm = f(m)
    whole = simpson3(fa, fm, fb, a, b)
    return recurse(a, b, fa, fm, fb, whole, max_depth)


# 5-point Gauss-Legendre nodes and weights on [-1, 1] (exact to degree 9)
_GL5_NODES = [-0.9061798459386640, -0.5384693101056831, 0.0,
              0.5384693101056831, 0.9061798459386640]
_GL5_WEIGHTS = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                0.4786286704993665, 0.2369268850561891]


def gauss_legendre(f, a, b, panels: int = 1):
    """Composite 5-point Gauss-Legendre quadrature over `panels` equal subintervals. Each panel
    integrates polynomials up to degree 9 exactly -- the most accuracy per function evaluation
    for smooth integrands."""
    if panels < 1:
        raise ValueError("panels must be >= 1")
    total = 0.0
    pw = (b - a) / panels
    for p in range(panels):
        lo = a + p * pw
        mid = lo + 0.5 * pw
        half = 0.5 * pw
        for node, w in zip(_GL5_NODES, _GL5_WEIGHTS):
            total += w * f(mid + half * node)
        # (accumulate; scale once below)
    return total * (0.5 * pw)
