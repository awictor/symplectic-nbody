"""The relativistic rocket: interstellar travel at constant 1 g.

A ship accelerating at a constant proper acceleration a (what the crew feels -- 1 g
gives Earth-like gravity) follows a hyperbolic worldline. The relativistic equations of
motion give clean closed forms in terms of the ship's own elapsed (proper) time tau:

    velocity      v(tau) = c tanh(a tau / c)
    Lorentz gamma        = cosh(a tau / c)
    distance      d(tau) = (c^2 / a) (cosh(a tau / c) - 1)
    Earth time    t(tau) = (c / a) sinh(a tau / c)

Because tanh saturates at 1, the ship approaches but never reaches c; meanwhile the
hyperbolic sines and cosines let proper time run far behind Earth time. At 1 g the
crew reaches ~0.77 c in a year of ship time, crosses to the galactic centre (~27,000
ly) in ~20 years of ship time (though ~27,000 yr pass on Earth), and could in principle
reach the Andromeda galaxy in ~28 years of proper time.

Fuel is the catch: the ideal (photon-drive) mass ratio to reach rapidity phi = a tau/c
and stop again is exp(2 phi) each way, which for a round trip to a nearby star already
demands astronomically more fuel than payload. This module gives velocity, gamma,
distance, Earth time, proper time to cover a distance, and the photon-rocket mass ratio,
and reproduces the 1-g reach across the Galaxy. SI units, years/light-years helpers.
Pure stdlib; the special-relativistic companion to the Oberth and beaming modules.
"""

from __future__ import annotations

import math

C = 2.99792458e8
G_EARTH = 9.80665
YEAR = 3.15576e7
LY = 9.4607e15


def velocity(a: float, tau: float) -> float:
    """Ship velocity (m/s) after proper time tau at proper acceleration a:
    v = c tanh(a tau / c). Approaches but never reaches c."""
    return C * math.tanh(a * tau / C)


def lorentz_gamma(a: float, tau: float) -> float:
    """Lorentz factor gamma = cosh(a tau / c) after proper time tau."""
    return math.cosh(a * tau / C)


def distance(a: float, tau: float) -> float:
    """Distance covered (m) in the launch frame: d = (c^2/a)(cosh(a tau/c) - 1)."""
    return C * C / a * (math.cosh(a * tau / C) - 1.0)


def earth_time(a: float, tau: float) -> float:
    """Elapsed time (s) in the launch (Earth) frame: t = (c/a) sinh(a tau / c).
    Runs far ahead of the ship's proper time for large tau."""
    return C / a * math.sinh(a * tau / C)


def proper_time_for_distance(a: float, d: float) -> float:
    """Ship proper time (s) to cover distance d from rest at acceleration a:
    tau = (c/a) arccosh(1 + a d / c^2). (Pure acceleration, no coasting/decel.)"""
    return C / a * math.acosh(1.0 + a * d / (C * C))


def photon_rocket_mass_ratio(a: float, tau: float) -> float:
    """Ideal photon-rocket initial/final mass ratio to reach rapidity phi = a tau/c:
    M0/M1 = exp(phi) for one acceleration leg (exp(2 phi) to accelerate then stop)."""
    phi = a * tau / C
    return math.exp(phi)


def rapidity(a: float, tau: float) -> float:
    """Rapidity phi = a tau / c: the additive 'relativistic velocity' whose tanh is
    v/c. Grows without bound even as v saturates at c."""
    return a * tau / C
