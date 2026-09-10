"""Tidal locking: why the Moon shows one face, and how long that takes.

A satellite raised into a tidal bulge by its primary does not respond instantly:
internal friction drags the bulge slightly out of line with the primary. That
misaligned bulge feels a torque that pushes the body's spin toward synchronous
rotation, where the same face points at the primary forever -- exactly what the
Moon does to Earth, Phobos to Mars, and Io/Europa/Ganymede to Jupiter.

The despinning time from an initial spin omega_i to synchronous scales as

    t_lock ~ (w * omega_i * a^6 * I * Q) / (3 * G * M_p^2 * k2 * R^5),

where a is the orbital distance, R and I the satellite's radius and moment of
inertia, M_p the primary mass, k2 the Love number (how easily the body deforms),
and Q the tidal quality factor (how lossy it is). The brutal a^6 dependence is the
whole story: locking time explodes with distance, so close-in bodies lock almost
immediately while distant ones never do within the age of the solar system.

This is why the Moon (a ~ 60 R_earth) locked long ago but the Earth, braking on the
Moon's far weaker tide, is only slowly spinning down (days lengthen ~1.8 ms/century);
why Mercury sits in a 3:2 spin-orbit resonance rather than fully locked; and why hot
Jupiters orbiting at a few stellar radii are all assumed synchronous. This module
gives the locking timescale, its steep scalings, and a boolean "locked within a time
budget," and reproduces the Moon-locked / Earth-unlocked contrast. SI units. Pure
stdlib; the dissipative sibling of the Roche and tidal-heating modules.
"""

from __future__ import annotations

import math

G = 6.67430e-11
YEAR = 3.15576e7
GYR = 1e9 * YEAR

M_EARTH = 5.972e24
M_MOON = 7.342e22
R_MOON = 1.7374e6
R_EARTH = 6.371e6
A_MOON = 3.844e8              # Earth-Moon distance
DAY = 86400.0

AGE_SOLAR_SYSTEM = 4.567 * GYR


def locking_time(omega_i: float, a: float, R: float, I: float, M_p: float,
                 k2: float = 0.3, Q: float = 100.0, w: float = 1.0) -> float:
    """Tidal despinning time (s) from initial spin omega_i to synchronous:

        t = w omega_i a^6 I Q / (3 G M_p^2 k2 R^5).

    k2 is the Love number, Q the tidal quality factor, w an order-unity structure
    constant. The a^6 factor dominates everything."""
    return w * omega_i * a ** 6 * I * Q / (3.0 * G * M_p ** 2 * k2 * R ** 5)


def moment_of_inertia(M: float, R: float, factor: float = 0.4) -> float:
    """Moment of inertia I = factor M R^2 (0.4 for a uniform sphere)."""
    return factor * M * R ** 2


def is_locked(omega_i: float, a: float, R: float, I: float, M_p: float,
              budget: float = AGE_SOLAR_SYSTEM, k2: float = 0.3,
              Q: float = 100.0, w: float = 1.0) -> bool:
    """True if the body locks within `budget` seconds (default the solar system's age)."""
    return locking_time(omega_i, a, R, I, M_p, k2, Q, w) <= budget


def max_locking_distance(omega_i: float, R: float, I: float, M_p: float,
                         budget: float = AGE_SOLAR_SYSTEM, k2: float = 0.3,
                         Q: float = 100.0, w: float = 1.0) -> float:
    """The orbital distance a (m) at which t_lock exactly equals the time budget.
    Inside this radius a body locks; outside it does not. Solve t ~ a^6 for a."""
    denom = w * omega_i * I * Q / (3.0 * G * M_p ** 2 * k2 * R ** 5)
    return (budget / denom) ** (1.0 / 6.0)
