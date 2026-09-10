"""The logistic map: the simplest equation that goes chaotic.

Take a population that grows but is capped by crowding, and iterate

    x_{n+1} = r x_n (1 - x_n),        0 <= x <= 1,

with r the growth rate. This one-line map is the textbook example of the route to chaos. As r
rises the long-term behaviour changes qualitatively:

    r < 1        : the population dies out (x -> 0)
    1 < r < 3    : it settles to one stable value 1 - 1/r
    3 < r < 3.449: it oscillates between two values (period 2)
    ... a cascade of period doublings (2, 4, 8, 16, ...) accumulating at
    r ~ 3.5699   : the onset of CHAOS, aperiodic and sensitive to initial conditions,
    r up to 4    : chaos interrupted by periodic windows (a famous period-3 window near 3.83).

The period-doublings arrive geometrically: the ratio of successive bifurcation spacings
converges to the universal Feigenbaum constant

    delta = 4.6692...,

which -- astonishingly -- is the same for ANY smooth map with a single quadratic maximum, not
just this one. It is a fingerprint of how chaos is born, seen in dripping taps, oscillating
chemical reactions, and electronic circuits alike. In the chaotic region the sensitivity to
initial conditions is measured by a positive Lyapunov exponent, the average log-stretching
per iteration.

This module gives one map step, the trajectory, the fixed point and its stability, the
long-term attractor set (the values a trajectory visits), the Lyapunov exponent, and the
Feigenbaum constant, and reproduces the 1-1/r fixed point, the period-2 onset at r=3, and
positive Lyapunov exponent in the chaotic regime. Pure stdlib; the nonlinear-dynamics
companion to the N-body Lyapunov and Poincare notes.
"""

from __future__ import annotations

import math

FEIGENBAUM_DELTA = 4.669201609   # universal period-doubling ratio
CHAOS_ONSET = 3.569945672        # r at the accumulation of period doublings


def step(x: float, r: float) -> float:
    """One iteration of the logistic map: x' = r x (1 - x)."""
    return r * x * (1.0 - x)


def trajectory(x0: float, r: float, n: int) -> list:
    """The first n iterates starting from x0 (list of length n+1 including x0)."""
    xs = [x0]
    x = x0
    for _ in range(n):
        x = step(x, r)
        xs.append(x)
    return xs


def fixed_point(r: float) -> float:
    """The nonzero fixed point x* = 1 - 1/r (exists and is the attractor for 1 < r < 3).
    Returns 0 for r <= 1 (extinction)."""
    if r <= 1.0:
        return 0.0
    return 1.0 - 1.0 / r


def fixed_point_stable(r: float) -> bool:
    """True if the nonzero fixed point is stable: |f'(x*)| = |2 - r| < 1, i.e. 1 < r < 3.
    At r=3 it loses stability and period-2 is born."""
    return 1.0 < r < 3.0


def attractor(r: float, x0: float = 0.5, transient: int = 2000, samples: int = 64) -> list:
    """The set of values the trajectory settles onto, after discarding `transient` iterates.
    Returns a sorted list of distinct attractor points (1 value = fixed point, 2 = period-2,
    many/continuous = chaos)."""
    x = x0
    for _ in range(transient):
        x = step(x, r)
    seen = []
    for _ in range(samples):
        x = step(x, r)
        seen.append(x)
    # collapse near-duplicates to count the period
    seen.sort()
    distinct = [seen[0]]
    for v in seen[1:]:
        if v - distinct[-1] > 1e-5:
            distinct.append(v)
    return distinct


def period(r: float, **kwargs) -> int:
    """Number of distinct points in the attractor (the period): 1 (fixed), 2, 4, ... or a
    large number in the chaotic regime."""
    return len(attractor(r, **kwargs))


def lyapunov_exponent(r: float, x0: float = 0.31, transient: int = 1000, n: int = 5000) -> float:
    """Lyapunov exponent lambda = <ln|f'(x)|> = <ln|r(1-2x)|> over the trajectory. Negative in
    the periodic windows (stable), positive in the chaotic regime (sensitive to initial
    conditions)."""
    x = x0
    for _ in range(transient):
        x = step(x, r)
    total = 0.0
    count = 0
    for _ in range(n):
        deriv = abs(r * (1.0 - 2.0 * x))
        if deriv > 1e-12:
            total += math.log(deriv)
            count += 1
        x = step(x, r)
    return total / count if count else 0.0


def is_chaotic(r: float, **kwargs) -> bool:
    """True if the Lyapunov exponent is positive -- the signature of chaos (sensitive
    dependence on initial conditions)."""
    return lyapunov_exponent(r, **kwargs) > 0.0
