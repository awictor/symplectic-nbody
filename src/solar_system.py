"""The real solar system, built from published orbital elements.

Units: AU, years, solar masses. In these units Newton's constant is
G = 4*pi^2 (since a 1 AU circular orbit around 1 solar mass has period 1 yr).

Planet masses and orbital elements (semi-major axis a, eccentricity e) are the
standard IAU/JPL values. Each planet is placed at perihelion on the +x axis with
the vis-viva speed for its (a, e), moving in +y -- a coplanar model that
reproduces the real orbital periods and lets Kepler's third law fall out of a
direct integration.
"""

from __future__ import annotations

import math
from typing import List

from nbody import NBody

G_AU = 4.0 * math.pi ** 2  # AU^3 / (Msun * yr^2)

# name: (mass [Msun], semi-major axis a [AU], eccentricity e)
PLANETS = {
    "Mercury": (1.6601e-7, 0.387098, 0.205630),
    "Venus":   (2.4478e-6, 0.723332, 0.006772),
    "Earth":   (3.0035e-6, 1.000000, 0.016710),
    "Mars":    (3.2271e-7, 1.523679, 0.093400),
    "Jupiter": (9.5479e-4, 5.204267, 0.048775),
    "Saturn":  (2.8589e-4, 9.582017, 0.055723),
    "Uranus":  (4.3662e-5, 19.18916, 0.047220),
    "Neptune": (5.1514e-5, 30.06992, 0.008590),
}

SUN_MASS = 1.0


def kepler_period(a: float, m_central: float = SUN_MASS) -> float:
    """Analytic orbital period T = 2*pi*sqrt(a^3 / (G*M)) in years."""
    return 2.0 * math.pi * math.sqrt(a ** 3 / (G_AU * m_central))


def build(planets: List[str] = None, include_sun: bool = True) -> NBody:
    """Construct an NBody solar-system model. Planets start at perihelion on the
    +x axis; the Sun is nudged so total momentum is zero (barycentric frame)."""
    if planets is None:
        planets = list(PLANETS.keys())

    masses = [SUN_MASS] if include_sun else []
    pos = [[0.0, 0.0, 0.0]] if include_sun else []
    vel = [[0.0, 0.0, 0.0]] if include_sun else []

    for name in planets:
        m, a, e = PLANETS[name]
        r_peri = a * (1.0 - e)
        # vis-viva relative to the Sun: v^2 = G*M*(2/r - 1/a)
        mu = G_AU * (SUN_MASS + m)
        v_peri = math.sqrt(mu * (2.0 / r_peri - 1.0 / a))
        masses.append(m)
        pos.append([r_peri, 0.0, 0.0])
        vel.append([0.0, v_peri, 0.0])

    body = NBody(masses=masses, pos=pos, vel=vel, G=G_AU)

    if include_sun:
        # remove net momentum so the barycentre stays put
        p = body.linear_momentum()
        for k in range(3):
            body.vel[0][k] -= p[k] / body.m[0]

    return body
