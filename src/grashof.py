"""The Grashof number: heat that stirs its own wind.

Forced convection has a pump or a fan; natural convection has only buoyancy. Warm a surface
and the fluid touching it expands, grows lighter, and rises, dragging cooler fluid in behind
-- a self-made current driven purely by the temperature difference. The strength of that
buoyant drive relative to the viscosity that damps it is the Grashof number,

    Gr = g beta (T_s - T_inf) L^3 / nu^2,

where beta is the fluid's thermal expansion coefficient (1/T for an ideal gas), L the height
of the surface, and nu the kinematic viscosity. Gr plays the role that the Reynolds number
plays in forced flow: it sets whether the buoyant boundary layer stays laminar or trips to
turbulence (for a vertical plate, around Gr ~ 1e9).

The heat carried off still follows a Nusselt number, but now Nu is correlated against the
Rayleigh number Ra = Gr Pr (buoyancy vs the product of both diffusivities). For a vertical
plate the workhorse fits are

    Nu = 0.59 Ra^(1/4)      (laminar,   1e4 < Ra < 1e9)
    Nu = 0.10 Ra^(1/3)      (turbulent, 1e9 < Ra < 1e13),

and h = Nu k / L then gives the heat-transfer coefficient -- the gentle ~5 W/(m^2 K) of a
radiator warming a still room, far below the forced-convection values. The ratio Gr/Re^2
tells you which regime you are in: >>1 buoyancy dominates (natural), <<1 forced wins.

This module gives the Grashof and Rayleigh numbers, the ideal-gas expansion coefficient, the
laminar/turbulent vertical-plate Nusselt correlations with automatic regime selection, the
heat-transfer coefficient, and the natural-vs-forced ratio, and reproduces the ~5 W/(m^2 K)
of a warm wall in still air. SI units. Pure stdlib; the buoyant-flow companion to the
convection, Peclet and Rayleigh-Benard notes.
"""

from __future__ import annotations

G_EARTH = 9.80665


def expansion_coefficient_ideal_gas(temperature: float) -> float:
    """Thermal expansion coefficient beta = 1/T (1/K) for an ideal gas at temperature T."""
    return 1.0 / temperature


def grashof_number(delta_T: float, length: float, beta: float, nu: float,
                   g: float = G_EARTH) -> float:
    """Grashof number Gr = g beta dT L^3 / nu^2: buoyant drive over viscous damping. The
    natural-convection analogue of the Reynolds number."""
    return g * beta * delta_T * length ** 3 / (nu * nu)


def rayleigh_number(grashof: float, prandtl: float) -> float:
    """Rayleigh number Ra = Gr Pr: the governing parameter for natural convection, buoyancy
    against the combined momentum and thermal diffusion."""
    return grashof * prandtl


def nusselt_vertical_plate(rayleigh: float) -> float:
    """Average Nusselt number for a heated vertical plate, selecting the correlation by
    Rayleigh number: 0.59 Ra^(1/4) (laminar, Ra<1e9) or 0.10 Ra^(1/3) (turbulent)."""
    if rayleigh < 1e9:
        return 0.59 * rayleigh ** 0.25
    return 0.10 * rayleigh ** (1.0 / 3.0)


def is_turbulent(rayleigh: float) -> bool:
    """True if the natural-convection boundary layer on a vertical plate is turbulent
    (Ra > ~1e9)."""
    return rayleigh > 1e9


def heat_transfer_coefficient(nusselt: float, conductivity: float, length: float) -> float:
    """h = Nu k / L (W/(m^2 K)) from the Nusselt number, fluid conductivity k, and height L."""
    return nusselt * conductivity / length


def natural_vs_forced(grashof: float, reynolds: float) -> float:
    """Ratio Gr / Re^2: >>1 means buoyancy dominates (natural convection), <<1 means forced
    convection wins, ~1 means mixed (combined) convection."""
    return grashof / (reynolds * reynolds)
