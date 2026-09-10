"""The Sedov-Taylor blast wave: a supernova remnant (or an atomic bomb).

A large amount of energy E released instantaneously into a uniform medium of
density rho drives a strong spherical shock. Once the swept-up mass dominates the
ejecta but radiative losses are still negligible, the flow is SELF-SIMILAR: the
only combination of E, rho, and time t with the dimensions of length is
(E t^2 / rho)^{1/5}, so the shock radius must be

    R(t) = xi0 (E t^2 / rho)^{1/5},

with a dimensionless constant xi0 ~ 1.15 for a gamma = 5/3 gas. Hence R ~ t^{2/5}
and the shock speed v = dR/dt = (2/5) R/t decelerates as t^{-3/5}.

G. I. Taylor famously ran this backwards: from a movie of the 1945 Trinity
fireball's radius-vs-time he estimated the bomb's yield (~20 kilotons) while it
was still classified. The same law sets the size and age of supernova remnants.

This module gives R(t), v(t), the shock temperature, and the inverse -- energy
from an observed radius and time. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

XI0 = 1.15266                # dimensionless Sedov constant for gamma = 5/3
K_B = 1.380649e-23           # Boltzmann constant
M_P = 1.6726219e-27          # proton mass


def shock_radius(E: float, rho: float, t: float, xi0: float = XI0) -> float:
    """Sedov-Taylor shock radius R(t) = xi0 (E t^2 / rho)^{1/5}."""
    return xi0 * (E * t * t / rho) ** 0.2


def shock_velocity(E: float, rho: float, t: float, xi0: float = XI0) -> float:
    """Shock speed v = dR/dt = (2/5) R/t."""
    return 0.4 * shock_radius(E, rho, t, xi0) / t


def energy_from_radius(R: float, rho: float, t: float, xi0: float = XI0) -> float:
    """Invert R(t) to recover the explosion energy E = rho R^5 / (xi0^5 t^2).
    This is exactly what Taylor did to the Trinity fireball."""
    return rho * R ** 5 / (xi0 ** 5 * t * t)


def shock_temperature(E: float, rho: float, t: float, gamma: float = 5.0 / 3.0,
                      mu: float = 0.6, xi0: float = XI0) -> float:
    """Post-shock temperature for a strong shock:
    T = 2(gamma-1)/(gamma+1)^2 * mu m_p v^2 / k_B  (immediate post-shock)."""
    v = shock_velocity(E, rho, t, xi0)
    return 2.0 * (gamma - 1.0) / (gamma + 1.0) ** 2 * mu * M_P * v * v / K_B


def swept_mass(E: float, rho: float, t: float, xi0: float = XI0) -> float:
    """Mass swept up into the shell, (4/3) pi R^3 rho."""
    R = shock_radius(E, rho, t, xi0)
    return 4.0 / 3.0 * math.pi * R ** 3 * rho
