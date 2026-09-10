"""J2 orbital precession: how a planet's bulge twists satellite orbits.

A perfectly spherical planet gives closed Keplerian ellipses. A real planet bulges at
its equator, and that oblateness -- captured by the dimensionless coefficient J2 -- adds
a perturbing potential that makes a satellite's orbit precess in two ways:

  * NODAL regression: the orbit plane's line of nodes drifts, at rate
        dOmega/dt = -(3/2) J2 (R/p)^2 n cos(i),
  * APSIDAL precession: the ellipse's major axis rotates in-plane, at rate
        domega/dt = (3/4) J2 (R/p)^2 n (4 - 5 sin^2 i) = ... (5 cos^2 i - 1),

with n the mean motion, R the planet's equatorial radius, p = a(1-e^2) the semi-latus
rectum, and i the inclination. For Earth J2 = 1.0826e-3, and a low orbit regresses its
nodes by several degrees per day.

This is not a nuisance but a design tool. Choosing the inclination so the nodal drift
exactly matches Earth's ~0.9856 deg/day motion around the Sun gives a SUN-SYNCHRONOUS
orbit -- the plane keeps a fixed angle to the Sun, so a satellite crosses the equator at
the same local time every pass (essential for imaging and weather satellites). Setting
5 cos^2 i - 2 = 0 (i = 63.4 deg) freezes the apsides -- the Molniya orbit that parks its
apogee over high latitudes.

This module gives the nodal and apsidal precession rates, the sun-synchronous
inclination, and the critical (frozen-apside) inclination, and reproduces the LEO nodal
drift and the 63.4-degree critical inclination. SI units, rates convertible to deg/day.
Pure stdlib; the perturbed-orbit companion to the Kepler and relativity modules.
"""

from __future__ import annotations

import math

G = 6.67430e-11
M_EARTH = 5.972e24
R_EARTH = 6.378137e6           # equatorial radius (m)
J2_EARTH = 1.08262668e-3
MU_EARTH = G * M_EARTH

DEG_PER_DAY = 180.0 / math.pi * 86400.0
SUN_RATE_DEG_DAY = 360.0 / 365.2422   # Earth's mean motion around the Sun


def mean_motion(a: float, mu: float = MU_EARTH) -> float:
    """Mean motion n = sqrt(mu / a^3) (rad/s)."""
    return math.sqrt(mu / a ** 3)


def nodal_precession_rate(a: float, e: float, i_rad: float,
                          J2: float = J2_EARTH, R: float = R_EARTH,
                          mu: float = MU_EARTH) -> float:
    """Nodal (line-of-nodes) regression rate dOmega/dt (rad/s):
    -(3/2) J2 (R/p)^2 n cos(i). Negative (westward) for prograde orbits."""
    n = mean_motion(a, mu)
    p = a * (1.0 - e * e)
    return -1.5 * J2 * (R / p) ** 2 * n * math.cos(i_rad)


def apsidal_precession_rate(a: float, e: float, i_rad: float,
                            J2: float = J2_EARTH, R: float = R_EARTH,
                            mu: float = MU_EARTH) -> float:
    """Apsidal (argument-of-perigee) precession rate domega/dt (rad/s):
    (3/4) J2 (R/p)^2 n (4 - 5 sin^2 i) = (3/4)... (5 cos^2 i - 1). Zero at the critical
    inclination i = 63.43 deg."""
    n = mean_motion(a, mu)
    p = a * (1.0 - e * e)
    return 0.75 * J2 * (R / p) ** 2 * n * (5.0 * math.cos(i_rad) ** 2 - 1.0)


def sun_synchronous_inclination(a: float, e: float = 0.0, J2: float = J2_EARTH,
                                R: float = R_EARTH, mu: float = MU_EARTH) -> float:
    """Inclination (radians) that makes the nodal regression match Earth's orbital
    rate around the Sun, giving a sun-synchronous orbit. Solve dOmega/dt = +omega_sun
    for cos(i) (yields a retrograde i > 90 deg)."""
    n = mean_motion(a, mu)
    p = a * (1.0 - e * e)
    omega_sun = SUN_RATE_DEG_DAY * math.pi / 180.0 / 86400.0   # rad/s
    cos_i = -omega_sun / (1.5 * J2 * (R / p) ** 2 * n)
    return math.acos(cos_i)


def critical_inclination() -> float:
    """The critical inclination (radians) where apsidal precession vanishes:
    5 cos^2 i - 1 = 0  ->  i = arccos(sqrt(1/5)) ~ 63.43 deg (the Molniya orbit)."""
    return math.acos(math.sqrt(1.0 / 5.0))
