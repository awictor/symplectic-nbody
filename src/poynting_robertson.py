"""Poynting-Robertson drag: why interplanetary dust spirals into the Sun.

A dust grain orbiting the Sun absorbs sunlight coming radially outward and re-emits
it isotropically in its own frame. But in the Sun's frame the grain is moving, so by
aberration the re-emitted photons carry away a little forward momentum -- a headwind
made of the grain's own thermal radiation. This Poynting-Robertson drag saps orbital
angular momentum and makes the grain spiral inward.

The inspiral time from a circular orbit of radius r for a grain of radius s and
density rho is

    t_PR = (c r^2) / (4 GM) * (1/beta) roughly,   with
    beta = (3 L) / (16 pi G M c rho s),

where beta is the famous ratio of radiation-pressure force to gravity (independent of
distance, since both go as 1/r^2). Writing it out,

    t_PR = 4 pi rho s c r^2 / (3 L) * ... ~ 400 (r/AU)^2 (rho/1000)(s/1e-6) years-ish.

Two consequences fall out. First, grains with beta > 1/2 are on unbound orbits the
instant they are released and get blown straight out as "beta meteoroids"; the
beta = 1 size is the blow-out radius. Second, everything smaller than a boulder but
bigger than blow-out size spirals in on timescales far shorter than the age of the
solar system, so the zodiacal dust we see must be continuously resupplied by comets
and asteroid collisions.

This module gives beta, the blow-out size, the PR inspiral time and its scalings, and
reproduces the ~micron-grain / ~10^4-year infall from 1 AU. SI units. Pure stdlib;
the radiation-drag companion to the blackbody and snow-line modules.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
L_SUN = 3.828e26
M_SUN = 1.989e30
AU = 1.495978707e11
YEAR = 3.15576e7

RHO_DUST = 3000.0              # typical silicate grain density (kg/m^3)


def beta(s: float, rho: float = RHO_DUST, L: float = L_SUN,
         M: float = M_SUN, Q_pr: float = 1.0) -> float:
    """Ratio of radiation-pressure force to gravity for a grain of radius s:
    beta = 3 L Q_pr / (16 pi G M c rho s). Distance-independent (both ~1/r^2)."""
    return 3.0 * L * Q_pr / (16.0 * math.pi * G * M * C * rho * s)


def blowout_size(rho: float = RHO_DUST, L: float = L_SUN, M: float = M_SUN,
                 Q_pr: float = 1.0) -> float:
    """Grain radius (m) at which beta = 1/2: released grains below this are unbound
    and blown out of the system ("beta meteoroids"). Solve beta(s) = 1/2."""
    return 3.0 * L * Q_pr / (16.0 * math.pi * G * M * C * rho * 0.5)


def is_blown_out(s: float, rho: float = RHO_DUST, L: float = L_SUN,
                 M: float = M_SUN, Q_pr: float = 1.0) -> bool:
    """True if a grain of radius s is on an unbound orbit when released (beta > 1/2)."""
    return beta(s, rho, L, M, Q_pr) > 0.5


def inspiral_time(r: float, s: float, rho: float = RHO_DUST, L: float = L_SUN,
                  M: float = M_SUN, Q_pr: float = 1.0) -> float:
    """Poynting-Robertson inspiral time (s) from a circular orbit of radius r to the
    star: t = c r^2 / (4 G M beta). Grows as r^2 and as grain size s."""
    b = beta(s, rho, L, M, Q_pr)
    return C * r * r / (4.0 * G * M * b)


def inspiral_time_years(r: float, s: float, rho: float = RHO_DUST,
                        L: float = L_SUN, M: float = M_SUN,
                        Q_pr: float = 1.0) -> float:
    """PR inspiral time in years."""
    return inspiral_time(r, s, rho, L, M, Q_pr) / YEAR
