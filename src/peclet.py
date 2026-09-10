"""The Peclet number: does stuff move by being carried, or by spreading?

Transport of heat or a dissolved species happens two ways at once -- it is *carried* by the
bulk flow (advection) and it *spreads* down its own gradient (diffusion). Which wins is set
by a single dimensionless ratio, the Peclet number:

    Pe = advective rate / diffusive rate = U L / D,

for a flow speed U over a length L with diffusivity D (mass diffusivity for a solute,
thermal diffusivity alpha = k/(rho c_p) for heat). Pe << 1 means diffusion dominates and the
concentration relaxes smoothly (a stirred-slowly cup, transport inside a cell); Pe >> 1 means
advection dominates and the substance is swept along in thin plumes and boundary layers
faster than it can spread (a river, a chimney, blood in an artery).

The Peclet number factors neatly into the flow and fluid properties:

    Pe_mass = Re * Sc,     Sc = nu / D        (Schmidt number)
    Pe_heat = Re * Pr,     Pr = nu / alpha     (Prandtl number)

where Re = U L / nu is the Reynolds number and nu the kinematic viscosity. The ratio of the
two diffusivities is the Lewis number Le = alpha / D = Sc / Pr, which decides whether heat or
mass diffuses faster and shapes flames and moist convection.

This module gives the Peclet number and its thermal/mass forms, the Prandtl, Schmidt and
Lewis numbers, thermal diffusivity from material properties, and the advection vs diffusion
crossover length, and reproduces water's Pr ~ 7, air's Pr ~ 0.7, and a solute's large
Schmidt number. SI units. Pure stdlib; the dimensionless-transport companion to the Reynolds
and Fick-diffusion notes.
"""

from __future__ import annotations


def peclet(velocity: float, length: float, diffusivity: float) -> float:
    """Peclet number Pe = U L / D. >>1: advection dominates (swept along); <<1: diffusion
    dominates (spreads smoothly); ~1: the crossover. Use mass diffusivity for a solute or
    thermal diffusivity alpha for heat."""
    return velocity * length / diffusivity


def thermal_diffusivity(conductivity: float, density: float, specific_heat: float) -> float:
    """Thermal diffusivity alpha = k / (rho c_p) (m^2/s): how fast a temperature disturbance
    spreads. ~1.4e-7 for water, ~2e-5 for air, ~1e-4 for metals."""
    return conductivity / (density * specific_heat)


def prandtl(kinematic_viscosity: float, thermal_diffusivity_: float) -> float:
    """Prandtl number Pr = nu / alpha: momentum diffusivity over thermal diffusivity. ~0.7
    for air, ~7 for water, <<1 for liquid metals, >>1 for oils. Sets the relative thickness
    of the velocity and thermal boundary layers."""
    return kinematic_viscosity / thermal_diffusivity_


def schmidt(kinematic_viscosity: float, mass_diffusivity: float) -> float:
    """Schmidt number Sc = nu / D: momentum diffusivity over mass diffusivity. ~1 for gases,
    ~1000 for a small molecule in water (momentum spreads far faster than the solute)."""
    return kinematic_viscosity / mass_diffusivity


def lewis(thermal_diffusivity_: float, mass_diffusivity: float) -> float:
    """Lewis number Le = alpha / D = Sc / Pr: thermal diffusivity over mass diffusivity.
    >1 means heat spreads faster than the species -- central to flames and moist air."""
    return thermal_diffusivity_ / mass_diffusivity


def peclet_from_reynolds(reynolds: float, prandtl_or_schmidt: float) -> float:
    """Peclet number as Pe = Re * Pr (heat) or Re * Sc (mass): the flow's Reynolds number
    times the fluid's diffusivity ratio."""
    return reynolds * prandtl_or_schmidt


def crossover_length(velocity: float, diffusivity: float) -> float:
    """Length at which advection and diffusion balance (Pe = 1): L = D / U (m). Below it
    diffusion wins, above it advection wins."""
    return diffusivity / velocity
