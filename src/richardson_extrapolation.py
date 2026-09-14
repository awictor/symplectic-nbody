"""Richardson extrapolation: squeezing high-order accuracy out of a low-order approximation.

Many numerical estimates A(h) approach the true value A as the step size h shrinks, with an error that
is a POWER SERIES in h: A(h) = A + c_1 h^p + c_2 h^{2p} + ... . The catch is that shrinking h alone is
slow and eventually loses to roundoff. RICHARDSON EXTRAPOLATION (1911) is the trick that turns a
crude, low-order method into a high-order one for free: combine estimates at two step sizes to CANCEL
the leading error term. If A(h) has leading error O(h^p), then

    A_improved = (t^p A(h/t) - A(h)) / (t^p - 1)

has error O(h^{2p}) -- the c_1 term is gone. Iterate the combination in a tableau and each column kills
the next error order, so a handful of cheap evaluations reach an accuracy that direct refinement would
never afford. This is the engine inside Romberg integration (extrapolating the trapezoid rule to
Gaussian accuracy), Bulirsch-Stoer ODE solving, and high-order numerical differentiation.

The classic application is the DERIVATIVE. The central difference (f(x+h) - f(x-h)) / (2h) has error
O(h^2) with only even powers, so Richardson extrapolation with p = 2 and t = 2 builds a tableau whose
diagonal converges as h^2, h^4, h^6, ... -- reaching near machine precision from a formula that on its
own is second order. It works identically for one-sided limits (lim_{h->0} g(h)) and any h-indexed
process.

This module implements the general Richardson tableau, specializations for numerical differentiation
and for limits, and the order-doubling error analysis. It is validated exactly: the derivative of
sin, exp, and polynomials matches the analytic value to near machine precision; the extrapolated
error is orders of magnitude smaller than the raw central difference; extrapolating a known
h-expansion (like (1+h)^{1/h} -> e) recovers the limit; the tableau reproduces Romberg on the
trapezoid ladder; and the empirical convergence order of the diagonal is the predicted 2, 4, 6.
Pure stdlib; the error-cancellation companion to the Romberg quadrature and finite-difference tools."""

from __future__ import annotations


def richardson_tableau(A, h0, p=1, t=2, levels=6):
    """General Richardson extrapolation tableau.

    A: a function A(h) whose error is O(h^p) + O(h^{2p}) + ...
    h0: initial step; t: refinement ratio (step divided by t each row); levels: tableau size.
    Returns the tableau (list of rows); the best estimate is tableau[-1][-1].
    """
    T = []
    h = h0
    for i in range(levels):
        row = [A(h)]
        for k in range(1, i + 1):
            factor = t ** (p * k)
            # combine the finer estimate row[k-1] with the coarser T[i-1][k-1] to cancel error order k
            val = (factor * row[k - 1] - T[i - 1][k - 1]) / (factor - 1)
            row.append(val)
        T.append(row)
        h /= t
    return T


def richardson_extrapolate(A, h0, p=1, t=2, levels=6):
    """Best Richardson-extrapolated estimate: the bottom-right corner of the tableau."""
    T = richardson_tableau(A, h0, p=p, t=t, levels=levels)
    return T[-1][-1]


def _central_difference(f, x, h):
    """Second-order central difference approximation to f'(x)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def derivative(f, x, h0=0.5, levels=6):
    """f'(x) by Richardson-extrapolating the central difference (error O(h^2), even powers).

    Uses p = 2 (only even powers of h appear in the central-difference error).
    """
    A = lambda h: _central_difference(f, x, h)
    return richardson_extrapolate(A, h0, p=2, t=2, levels=levels)


def derivative_tableau(f, x, h0=0.5, levels=6):
    """The full Richardson tableau for the derivative (diagonal converges as h^2, h^4, h^6, ...)."""
    A = lambda h: _central_difference(f, x, h)
    return richardson_tableau(A, h0, p=2, t=2, levels=levels)


def limit(g, h0=1.0, p=1, t=2, levels=8):
    """Estimate lim_{h->0} g(h) by Richardson extrapolation (default leading order p = 1)."""
    return richardson_extrapolate(g, h0, p=p, t=t, levels=levels)


def second_derivative(f, x, h0=0.5, levels=6):
    """f''(x) by Richardson-extrapolating the second central difference (error O(h^2))."""
    A = lambda h: (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    return richardson_extrapolate(A, h0, p=2, t=2, levels=levels)


def convergence_order(A, h0, p, t, levels):
    """Empirical convergence order of the tableau diagonal against a reference (bottom-right).

    Returns the sequence of estimated orders between successive diagonal entries.
    """
    import math
    T = richardson_tableau(A, h0, p=p, t=t, levels=levels)
    best = T[-1][-1]
    diag = [T[i][i] for i in range(levels)]
    errs = [abs(d - best) for d in diag[:-1]]
    orders = []
    for i in range(len(errs) - 1):
        if errs[i] > 0 and errs[i + 1] > 0:
            orders.append(math.log(errs[i] / errs[i + 1]) / math.log(t))
    return orders
