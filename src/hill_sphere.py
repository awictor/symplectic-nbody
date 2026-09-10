"""The Hill sphere: how far a planet's gravity keeps its moons.

A moon orbiting a planet feels two pulls: the planet holding it in, and the parent star
trying to steal it away. The planet wins inside its Hill sphere, of radius

    r_H = a (1 - e) (m / (3 M))^(1/3),

with a the planet's orbital semi-major axis (times 1-e for eccentric orbits), m the
planet's mass, and M the star's. Beyond r_H, solar tides overpower the planet and an
orbit is not stable; in practice moons survive only out to ~1/2-1/3 r_H for prograde
orbits.

For Earth (a = 1 AU, m = 3e-6 M_sun) the Hill radius is ~1.5 million km, about four
times the Moon's distance -- comfortably bound. The Moon's own Hill sphere (~60,000 km)
is why it holds no sub-moons. A planet closer to its star has a smaller Hill sphere
(hot Jupiters can barely keep moons), and the concept doubles as the feeding zone from
which a forming planet gathers material and the spacing that keeps planetary orbits
mutually stable (the "mutual Hill radius").

This module gives the Hill radius, the practical stable-moon limit, the mutual Hill
radius between two planets, and a moon-stability check, and reproduces Earth's ~1.5
million km Hill sphere. SI units. Pure stdlib; the three-body-stability companion to
the CR3BP and Roche modules.
"""

from __future__ import annotations

import math

AU = 1.495978707e11
M_SUN = 1.989e30
M_EARTH = 5.972e24
R_EARTH = 6.371e6

# moons are practically stable only out to ~this fraction of r_H (prograde)
STABLE_FRACTION = 0.5


def hill_radius(a: float, m: float, M: float, e: float = 0.0) -> float:
    """Hill radius r_H = a (1-e) (m / 3M)^(1/3) (m). a the orbital semi-major axis,
    m the secondary (planet) mass, M the primary (star) mass."""
    return a * (1.0 - e) * (m / (3.0 * M)) ** (1.0 / 3.0)


def stable_moon_limit(a: float, m: float, M: float, e: float = 0.0,
                      fraction: float = STABLE_FRACTION) -> float:
    """Practical outer limit (m) for a stable prograde moon orbit: ~1/2 r_H."""
    return fraction * hill_radius(a, m, M, e)


def is_moon_stable(moon_distance: float, a: float, m: float, M: float,
                   e: float = 0.0, fraction: float = STABLE_FRACTION) -> bool:
    """True if a moon at `moon_distance` from the planet lies inside the practical
    stability limit (~1/2 the Hill radius)."""
    return moon_distance < stable_moon_limit(a, m, M, e, fraction)


def mutual_hill_radius(a1: float, a2: float, m1: float, m2: float,
                       M: float) -> float:
    """Mutual Hill radius of two planets: R_H = ((m1+m2)/3M)^(1/3) (a1+a2)/2 (m).
    Orbital separations of many mutual Hill radii are needed for long-term stability."""
    return ((m1 + m2) / (3.0 * M)) ** (1.0 / 3.0) * 0.5 * (a1 + a2)


def separation_in_mutual_hill(a1: float, a2: float, m1: float, m2: float,
                              M: float) -> float:
    """Orbital separation (a2 - a1) expressed in mutual Hill radii -- the standard
    dynamical-spacing measure (systems need roughly > 10 for stability)."""
    return (a2 - a1) / mutual_hill_radius(a1, a2, m1, m2, M)
