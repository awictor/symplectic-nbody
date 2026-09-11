"""Horner's method: evaluating a polynomial the fast, stable way.

Evaluating a_n x^n + ... + a_1 x + a_0 by computing each power x^k separately costs about 2n
multiplications and, worse, sums terms of wildly different magnitudes -- numerically shaky.
Horner's method (published 1819, known to Newton and earlier) rewrites the polynomial as nested
multiplication,

    p(x) = (...((a_n x + a_{n-1}) x + a_{n-2}) x + ... ) x + a_0,

evaluating it in exactly n multiplications and n additions, left to right, with far better
rounding behaviour. It is the standard way every math library evaluates a polynomial.

The same nested sweep is SYNTHETIC DIVISION: dividing p(x) by (x - r) drops out of Horner as a
by-product -- the intermediate values ARE the quotient's coefficients, and the final value is
the remainder p(r) (the Remainder Theorem). Running Horner a second time on the quotient gives
p'(r), the derivative, for free. Together these make Horner the engine of Newton's method for
polynomial roots: p(r) and p'(r) in two linear sweeps, then r <- r - p(r)/p'(r).

This module evaluates polynomials by Horner (coefficients highest-degree first), does synthetic
division, computes derivatives, and finds real roots by Newton refinement, checking everything
against direct power-sum evaluation. Pure stdlib; the polynomial companion to the Kahan-summation
note.
"""

from __future__ import annotations


def horner(coeffs, x):
    """Evaluate the polynomial with coefficients `coeffs` (highest degree first) at x, using
    Horner's nested multiplication in n mults and n adds. horner([2,-6,2,-1], 3) = 2*27-6*9+2*3-1."""
    result = 0.0
    for a in coeffs:
        result = result * x + a
    return result


def direct_eval(coeffs, x):
    """Reference: evaluate by summing a_k x^(deg-k) with explicit powers. Slower and less
    stable; used to check Horner."""
    n = len(coeffs)
    return sum(a * x ** (n - 1 - i) for i, a in enumerate(coeffs))


def synthetic_division(coeffs, r):
    """Divide the polynomial by (x - r). Returns (quotient_coeffs, remainder), where the
    quotient has one lower degree and the remainder equals p(r) (the Remainder Theorem)."""
    if not coeffs:
        return [], 0.0
    quotient = [coeffs[0]]
    for a in coeffs[1:]:
        quotient.append(quotient[-1] * r + a)
    remainder = quotient.pop()          # the last value is p(r), not a quotient coefficient
    return quotient, remainder


def evaluate_with_derivative(coeffs, x):
    """Return (p(x), p'(x)) in two Horner sweeps -- the value and derivative for the price of
    a little bookkeeping (used by Newton's method)."""
    p = 0.0     # running p(x)
    dp = 0.0    # running p'(x)
    for a in coeffs:
        dp = dp * x + p     # derivative recurrence (uses the previous p)
        p = p * x + a
    return p, dp


def derivative_coeffs(coeffs):
    """The coefficients of p'(x) (highest degree first). d/dx of a_n x^n + ... is
    n a_n x^{n-1} + ..."""
    n = len(coeffs) - 1
    if n <= 0:
        return [0.0]
    return [coeffs[i] * (n - i) for i in range(n)]


def newton_root(coeffs, x0, tol: float = 1e-12, max_iter: int = 100):
    """Refine a real root of the polynomial from the initial guess x0 by Newton's method, using
    Horner to get p and p' each step. Returns (root, iterations) or raises if the derivative
    vanishes or it fails to converge."""
    x = float(x0)
    for it in range(1, max_iter + 1):
        p, dp = evaluate_with_derivative(coeffs, x)
        if dp == 0.0:
            raise ValueError("zero derivative; Newton step undefined")
        step = p / dp
        x -= step
        if abs(step) <= tol * (1.0 + abs(x)):
            return x, it
    raise ValueError("Newton's method did not converge")


def deflate(coeffs, root):
    """Divide out a known root, returning the lower-degree polynomial (the quotient of synthetic
    division by (x - root)). Used to find successive roots."""
    quotient, _ = synthetic_division(coeffs, root)
    return quotient


def real_roots(coeffs, guesses=None, tol: float = 1e-10):
    """Find real roots by Newton refinement plus deflation. `guesses` supplies starting points
    (defaults to a spread of values); returns the sorted distinct real roots found. This is a
    simple educational solver, not a robust general root-finder."""
    roots = []
    poly = [float(c) for c in coeffs]
    # strip leading zeros
    while len(poly) > 1 and poly[0] == 0.0:
        poly = poly[1:]
    starts = guesses if guesses is not None else [-10, -3, -1, -0.5, 0.5, 1, 3, 10]
    while len(poly) > 2:                # while degree >= 2, peel off one root
        found = None
        for g in starts:
            try:
                r, _ = newton_root(poly, g)
            except ValueError:
                continue
            if abs(horner(poly, r)) < 1e-6:
                found = r
                break
        if found is None:
            break
        roots.append(found)
        poly = deflate(poly, found)
    if len(poly) == 2 and poly[0] != 0.0:   # linear: a x + b -> -b/a
        roots.append(-poly[1] / poly[0])
    # deduplicate close roots
    roots.sort()
    distinct = []
    for r in roots:
        if not distinct or abs(r - distinct[-1]) > tol * (1 + abs(r)):
            distinct.append(r)
    return distinct
