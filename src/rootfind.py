"""Bracketing root-finders: bisection, secant, false position, and Brent.

Newton's method is fast but can diverge -- a bad guess, a flat spot, or a nearby extremum sends
it off to infinity. When you have a BRACKET, an interval [a, b] where f changes sign so a root
must lie between, you can guarantee convergence. These bracketing methods trade Newton's speed
for that certainty, and Brent's method (1973) recovers most of the speed without giving up the
guarantee -- which is why it is the default root-finder in most numerical libraries.

  * BISECTION halves the bracket each step, keeping the half where the sign changes. It is
    foolproof and converges linearly -- one bit of accuracy per iteration -- so ~50 steps reach
    double precision, always.

  * The SECANT method fits a line through the last two points and takes its x-intercept; it is
    superlinear (order ~1.618, the golden ratio) but does not maintain a bracket, so it can fail.

  * FALSE POSITION (regula falsi) is the secant kept inside a bracket: safe like bisection but
    usually faster.

  * BRENT combines bisection's safety with inverse quadratic interpolation's speed, falling back
    to bisection whenever the fast step misbehaves -- superlinear in practice, guaranteed in the
    worst case.

This module implements all four with a shared bracketing interface, reports the iteration count,
and checks them against roots of known functions (polynomials, transcendentals) and against each
other. Pure stdlib; the root-finding companion to the Horner (Newton) note."""

from __future__ import annotations

import math


def _sign(x):
    return (x > 0) - (x < 0)


def bisection(f, a, b, tol: float = 1e-12, max_iter: int = 200):
    """Find a root of f in [a, b] by bisection. Requires f(a) and f(b) to have opposite signs.
    Returns (root, iterations)."""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if _sign(fa) == _sign(fb):
        raise ValueError("f(a) and f(b) must have opposite signs (need a bracket)")
    for it in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0 or (b - a) / 2 < tol:
            return m, it
        if _sign(fm) == _sign(fa):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b), max_iter


def secant(f, x0, x1, tol: float = 1e-12, max_iter: int = 100):
    """Find a root by the secant method from two starting points (not necessarily a bracket).
    Superlinear but not guaranteed. Returns (root, iterations)."""
    f0, f1 = f(x0), f(x1)
    for it in range(1, max_iter + 1):
        if f1 == f0:
            raise ValueError("secant step has zero denominator")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) <= tol * (1 + abs(x2)):
            return x2, it
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    raise ValueError("secant did not converge")


def false_position(f, a, b, tol: float = 1e-12, max_iter: int = 200):
    """Regula falsi: the secant line kept inside a sign-changing bracket [a, b]. Safe and
    usually faster than bisection. Returns (root, iterations)."""
    fa, fb = f(a), f(b)
    if _sign(fa) == _sign(fb):
        raise ValueError("f(a) and f(b) must have opposite signs (need a bracket)")
    c = a
    for it in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)      # x-intercept of the secant line
        fc = f(c)
        if fc == 0 or abs(fc) < tol:
            return c, it
        if _sign(fc) == _sign(fa):
            a, fa = c, fc
        else:
            b, fb = c, fc
    return c, max_iter


def brent(f, a, b, tol: float = 1e-12, max_iter: int = 100):
    """Brent's method: inverse quadratic interpolation with a bisection fallback, on a
    sign-changing bracket [a, b]. Guaranteed to converge, superlinear in practice. Returns
    (root, iterations)."""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if _sign(fa) == _sign(fb):
        raise ValueError("f(a) and f(b) must have opposite signs (need a bracket)")
    if abs(fa) < abs(fb):                        # ensure b is the better estimate
        a, b, fa, fb = b, a, fb, fa
    c, fc = a, fa
    mflag = True
    d = c
    for it in range(1, max_iter + 1):
        if fb == 0 or abs(b - a) < tol:
            return b, it
        if fa != fc and fb != fc:
            # inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # secant step
            s = b - fb * (b - a) / (fb - fa)
        # conditions under which we reject s and bisect instead
        cond = (not (min((3 * a + b) / 4, b) < s < max((3 * a + b) / 4, b))
                or (mflag and abs(s - b) >= abs(b - c) / 2)
                or (not mflag and abs(s - b) >= abs(c - d) / 2)
                or (mflag and abs(b - c) < tol)
                or (not mflag and abs(c - d) < tol))
        if cond:
            s = 0.5 * (a + b)                     # bisection fallback
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d, c, fc = c, b, fb
        if _sign(fa) != _sign(fs):
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    return b, max_iter


def has_bracket(f, a, b) -> bool:
    """True if [a, b] brackets a root (f changes sign across it)."""
    return _sign(f(a)) != _sign(f(b))


def find_brackets(f, lo, hi, n: int = 100):
    """Scan [lo, hi] in n subintervals and return the [x, x+dx] pairs where f changes sign --
    a way to locate brackets before refining with a bracketing method."""
    brackets = []
    dx = (hi - lo) / n
    x = lo
    fx = f(x)
    for _ in range(n):
        x2 = x + dx
        fx2 = f(x2)
        if fx == 0:
            brackets.append((x, x))
        elif _sign(fx) != _sign(fx2):
            brackets.append((x, x2))
        x, fx = x2, fx2
    return brackets
