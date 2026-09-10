"""Rayleigh-Benard convection: when a heated fluid layer starts to churn.

Heat a fluid layer from below and two effects compete. Buoyancy wants to lift the
warm, less-dense fluid at the bottom; viscosity and thermal diffusion want to smear
the temperature difference away before any parcel can rise. The Rayleigh number is
the ratio of the destabilizing buoyancy to the stabilizing diffusion,

    Ra = g alpha dT d^3 / (nu kappa),

with g gravity, alpha the thermal expansion coefficient, dT the top-to-bottom
temperature difference, d the layer depth, nu the kinematic viscosity, and kappa the
thermal diffusivity. Below a critical value the layer just conducts heat; above it,
convection switches on abruptly in a pattern of rolls or hexagonal cells.

For a layer between two rigid, no-slip plates the critical Rayleigh number is a pure
number, Ra_c ~ 1708 -- one of the cleanest predictions in fluid dynamics, confirmed to
better than a percent. Once Ra exceeds Ra_c the fluid carries far more heat than
conduction alone; the enhancement is the Nusselt number, which for turbulent
convection follows roughly Nu ~ (Ra / Ra_c)^(1/3).

The same instability drives granulation on the Sun, mantle convection that moves
continents, and the cells in a pot of miso soup -- everywhere a fluid is heated from
below. This module gives the Rayleigh number, the onset verdict against Ra_c, the
Nusselt heat enhancement, and the critical temperature difference, and reproduces the
Ra_c ~ 1708 onset. SI units. Pure stdlib; the thermal-convection companion to the
Brunt-Vaisala and atmosphere modules.
"""

from __future__ import annotations

import math

G_EARTH = 9.80665
RA_CRITICAL_RIGID = 1707.762    # rigid-rigid boundaries (Rayleigh's classic value)
RA_CRITICAL_FREE = 657.511      # free-free (stress-free) boundaries, (27/4) pi^4


def rayleigh_number(dT: float, d: float, alpha: float, nu: float,
                    kappa: float, g: float = G_EARTH) -> float:
    """Rayleigh number Ra = g alpha dT d^3 / (nu kappa). alpha is the thermal
    expansion coefficient (1/K), nu kinematic viscosity (m^2/s), kappa thermal
    diffusivity (m^2/s), d the layer depth (m), dT the temperature drop (K)."""
    return g * alpha * dT * d ** 3 / (nu * kappa)


def is_convecting(Ra: float, Ra_c: float = RA_CRITICAL_RIGID) -> bool:
    """True if the layer convects: Ra exceeds the critical Rayleigh number."""
    return Ra > Ra_c


def critical_delta_T(d: float, alpha: float, nu: float, kappa: float,
                     g: float = G_EARTH, Ra_c: float = RA_CRITICAL_RIGID) -> float:
    """Temperature difference (K) at the onset of convection: the dT that makes
    Ra = Ra_c. Below it the layer only conducts."""
    return Ra_c * nu * kappa / (g * alpha * d ** 3)


def nusselt_number(Ra: float, Ra_c: float = RA_CRITICAL_RIGID,
                   exponent: float = 1.0 / 3.0) -> float:
    """Nusselt number Nu = total heat / conductive heat. Nu=1 below onset; above it
    Nu ~ (Ra/Ra_c)^exponent (the 1/3 law of hard turbulent convection)."""
    if Ra <= Ra_c:
        return 1.0
    return (Ra / Ra_c) ** exponent


def thermal_diffusivity(k: float, rho: float, c_p: float) -> float:
    """Thermal diffusivity kappa = k / (rho c_p) (m^2/s) from conductivity k,
    density rho and specific heat c_p."""
    return k / (rho * c_p)
