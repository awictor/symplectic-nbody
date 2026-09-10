"""The Jeans instability: when a gas cloud collapses to form stars.

A self-gravitating gas cloud is a tug of war between pressure (which resists
compression, communicated at the sound speed c_s) and gravity (which pulls it
together). Linearizing the fluid + Poisson equations for a plane-wave
perturbation ~ exp(i(k x - omega t)) gives the dispersion relation

    omega^2 = c_s^2 k^2 - 4 pi G rho0.

  * SHORT wavelengths (large k): omega^2 > 0, real omega -- the perturbation is a
    stable, oscillating SOUND WAVE; pressure wins.
  * LONG wavelengths (small k): omega^2 < 0, imaginary omega -- the perturbation
    grows exponentially: the cloud COLLAPSES; gravity wins.

The crossover is the Jeans wavenumber k_J = sqrt(4 pi G rho0)/c_s, i.e. the
Jeans length lambda_J = 2 pi / k_J. A cloud larger than lambda_J (more massive
than the Jeans mass M_J ~ rho lambda_J^3) is unstable and forms stars. This is
the threshold criterion behind all star and structure formation.

This module gives the dispersion relation, the Jeans length/mass, the growth
rate of unstable modes, and the free-fall time. Pure stdlib; generic units (set G).
"""

from __future__ import annotations

import math
from typing import Tuple


def jeans_wavenumber(cs: float, rho0: float, G: float = 1.0) -> float:
    """k_J = sqrt(4 pi G rho0) / c_s. Modes with k < k_J are unstable."""
    return math.sqrt(4.0 * math.pi * G * rho0) / cs


def jeans_length(cs: float, rho0: float, G: float = 1.0) -> float:
    """lambda_J = 2 pi / k_J = c_s sqrt(pi / (G rho0))."""
    return 2.0 * math.pi / jeans_wavenumber(cs, rho0, G)


def jeans_mass(cs: float, rho0: float, G: float = 1.0) -> float:
    """A mass estimate: the gas within a sphere of radius lambda_J/2,
    M_J = (4/3) pi (lambda_J/2)^3 rho0."""
    lam = jeans_length(cs, rho0, G)
    return 4.0 / 3.0 * math.pi * (0.5 * lam) ** 3 * rho0


def omega_squared(k: float, cs: float, rho0: float, G: float = 1.0) -> float:
    """The dispersion relation omega^2 = c_s^2 k^2 - 4 pi G rho0.
    Positive -> stable sound wave; negative -> gravitational collapse."""
    return cs * cs * k * k - 4.0 * math.pi * G * rho0


def is_unstable(k: float, cs: float, rho0: float, G: float = 1.0) -> bool:
    """True if a perturbation of wavenumber k grows (collapses)."""
    return omega_squared(k, cs, rho0, G) < 0.0


def growth_rate(k: float, cs: float, rho0: float, G: float = 1.0) -> float:
    """For an unstable mode, the exponential growth rate
    gamma = sqrt(4 pi G rho0 - c_s^2 k^2); returns 0 for stable modes."""
    w2 = omega_squared(k, cs, rho0, G)
    return math.sqrt(-w2) if w2 < 0.0 else 0.0


def free_fall_time(rho0: float, G: float = 1.0) -> float:
    """Gravitational free-fall (collapse) time t_ff = sqrt(3 pi / (32 G rho0)).
    This is the timescale on which a Jeans-unstable, pressureless cloud collapses,
    and the k->0 growth rate approaches sqrt(4 pi G rho) = a few / t_ff."""
    return math.sqrt(3.0 * math.pi / (32.0 * G * rho0))
