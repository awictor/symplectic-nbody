"""Hawking radiation and black-hole thermodynamics.

A black hole is not truly black: quantum effects at the horizon make it radiate
like a body of temperature

    T_H = hbar c^3 / (8 pi G M k_B),

INVERSELY proportional to mass -- big holes are colder. It carries the
Bekenstein-Hawking entropy

    S = k_B c^3 A / (4 G hbar) = 4 pi k_B G M^2 / (hbar c),

one quarter of the horizon area in Planck units. Radiating drains its mass; a
solitary hole evaporates completely in a time that scales as M^3,

    t_evap = 5120 pi G^2 M^3 / (hbar c^4).

Consequences this module reproduces from fundamental constants:
  * a solar-mass hole is at ~60 nanokelvin and lives ~10^67 years (utterly cold
    and effectively eternal);
  * a primordial hole of ~10^11-10^12 kg is evaporating in a Hubble time NOW,
    ending in a burst;
  * the entropy of a stellar black hole dwarfs every other entropy in physics.

SI units. Pure stdlib.
"""

from __future__ import annotations

import math

HBAR = 1.054571817e-34
C = 2.99792458e8
G = 6.67430e-11
K_B = 1.380649e-23
M_SUN = 1.98892e30
YEAR = 3.15576e7
# Planck length^2 = G hbar / c^3
L_P2 = G * HBAR / C ** 3


def hawking_temperature(M: float) -> float:
    """T_H = hbar c^3 / (8 pi G M k_B), in kelvin."""
    return HBAR * C ** 3 / (8.0 * math.pi * G * M * K_B)


def schwarzschild_radius(M: float) -> float:
    """r_s = 2 G M / c^2."""
    return 2.0 * G * M / C ** 2


def horizon_area(M: float) -> float:
    """A = 4 pi r_s^2."""
    return 4.0 * math.pi * schwarzschild_radius(M) ** 2


def bekenstein_hawking_entropy(M: float) -> float:
    """S = k_B A / (4 l_p^2), in J/K (i.e. S/k_B is the dimensionless count)."""
    return K_B * horizon_area(M) / (4.0 * L_P2)


def entropy_over_kb(M: float) -> float:
    """Dimensionless entropy S/k_B = A / (4 l_p^2) = 4 pi G M^2 / (hbar c)."""
    return 4.0 * math.pi * G * M * M / (HBAR * C)


def evaporation_time(M: float) -> float:
    """Total evaporation time (seconds): t = 5120 pi G^2 M^3 / (hbar c^4)."""
    return 5120.0 * math.pi * G ** 2 * M ** 3 / (HBAR * C ** 4)


def mass_evaporating_in(t: float) -> float:
    """Invert t_evap(M) = t: the mass whose lifetime equals t seconds.
    M = ( t hbar c^4 / (5120 pi G^2) )^{1/3}."""
    return (t * HBAR * C ** 4 / (5120.0 * math.pi * G ** 2)) ** (1.0 / 3.0)


def mass_loss_rate(M: float) -> float:
    """dM/dt < 0 (kg/s), from dE/dt = -sigma A T^4 / c^2 with the Stefan-Boltzmann
    law. Equivalent (up to greybody factors) to -M / (3 t_evap)."""
    return -M / (3.0 * evaporation_time(M))
