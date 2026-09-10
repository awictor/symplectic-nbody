"""Escape and the cosmic velocities: the speeds to leave Earth, Sun, Galaxy.

Several thresholds govern how fast you must go to leave a gravitating body:

  * ORBITAL (first cosmic) speed:  v1 = sqrt(G M / r)   -- circular orbit at r.
  * ESCAPE (second cosmic) speed:  v2 = sqrt(2 G M / r) = sqrt(2) v1
    -- the speed to reach infinity with zero energy left over.
  * THIRD cosmic speed: leave the Solar System from Earth's orbit (combining
    escape from the Sun with Earth's orbital motion).

From Earth's surface v1 ~ 7.9 km/s (low orbit) and v2 ~ 11.2 km/s (escape); from
the Solar System near Earth ~42 km/s heliocentric. The escape speed also sets a
black hole's boundary: setting v2 = c gives the Schwarzschild radius. This module
computes each speed and reproduces the textbook values. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
M_EARTH = 5.972e24
R_EARTH = 6.371e6
M_SUN = 1.98892e30
AU = 1.495978707e11


def orbital_speed(M: float, r: float) -> float:
    """First cosmic speed: circular-orbit speed v1 = sqrt(G M / r)."""
    return math.sqrt(G * M / r)


def escape_speed(M: float, r: float) -> float:
    """Second cosmic speed: escape speed v2 = sqrt(2 G M / r)."""
    return math.sqrt(2.0 * G * M / r)


def escape_over_orbital() -> float:
    """The universal ratio v_escape / v_orbital = sqrt(2)."""
    return math.sqrt(2.0)


def solar_system_escape_from_earth_orbit() -> float:
    """Heliocentric escape speed at Earth's orbital radius (sqrt(2) times Earth's
    orbital speed about the Sun) -- the speed needed to leave the Solar System."""
    return escape_speed(M_SUN, AU)


def schwarzschild_radius_from_escape(M: float) -> float:
    """Radius where the escape speed equals c: setting v2 = c in
    sqrt(2 G M / r) = c gives r = 2 G M / c^2 -- the Schwarzschild radius."""
    return 2.0 * G * M / (C * C)
