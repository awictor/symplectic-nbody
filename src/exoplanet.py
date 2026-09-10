"""Exoplanet detection: transits and radial-velocity wobbles.

Two methods find most exoplanets, and both are simple geometry + Kepler:

TRANSIT. When a planet crosses its star's disk, it blocks a fraction of the light
equal to the area ratio,

    depth = (R_planet / R_star)^2.

Jupiter across the Sun dims it by ~1% (0.01), Earth by ~0.008% (8.4e-5) -- the
tiny dip Kepler and TESS hunt for. The transit lasts roughly the time the planet
takes to cross the stellar diameter on its orbit.

RADIAL VELOCITY. The planet and star orbit their common barycentre, so the star
wobbles. Its line-of-sight velocity semi-amplitude is

    K = (2 pi G / P)^{1/3} * m_p sin i / (M_star + m_p)^{2/3} / sqrt(1 - e^2).

Jupiter makes the Sun wobble by ~12.5 m/s over 12 years; Earth by only ~0.09 m/s
-- the precision radial-velocity surveys must reach. This module gives the
transit depth, the transit duration, and the RV semi-amplitude, and reproduces
those numbers. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
M_SUN = 1.98892e30
R_SUN = 6.957e8
M_JUP = 1.898e27
R_JUP = 6.9911e7
M_EARTH = 5.972e24
R_EARTH = 6.371e6
AU = 1.495978707e11
YEAR = 3.15576e7


def transit_depth(R_planet: float, R_star: float) -> float:
    """Fractional dip in brightness during transit: (R_p / R_star)^2."""
    return (R_planet / R_star) ** 2


def orbital_period(a: float, M_star: float) -> float:
    """Kepler period P = 2 pi sqrt(a^3 / (G M_star)) (s)."""
    return 2.0 * math.pi * math.sqrt(a ** 3 / (G * M_star))


def transit_duration(a: float, M_star: float, R_star: float) -> float:
    """Approximate central-transit duration: the time to cross the stellar
    diameter at the orbital speed, T = (P / pi)(R_star / a)."""
    P = orbital_period(a, M_star)
    return (P / math.pi) * (R_star / a)


def rv_semi_amplitude(m_planet: float, M_star: float, a: float,
                      e: float = 0.0, inclination: float = math.pi / 2) -> float:
    """Stellar radial-velocity semi-amplitude K (m/s):
    K = (2 pi G / P)^{1/3} m_p sin i / (M_star + m_p)^{2/3} / sqrt(1-e^2)."""
    P = orbital_period(a, M_star + m_planet)
    return ((2.0 * math.pi * G / P) ** (1.0 / 3.0)
            * m_planet * math.sin(inclination)
            / (M_star + m_planet) ** (2.0 / 3.0)
            / math.sqrt(1.0 - e * e))
