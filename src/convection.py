"""Convective heat transfer: how fast a moving fluid carries heat off a surface.

When a fluid flows past a hot surface it strips heat away far faster than still conduction,
and the rate is captured by Newton's law of cooling,

    q = h (T_surface - T_fluid),

with h the convective heat-transfer coefficient (W/(m^2 K)). The whole art is finding h, and
it is set by a dimensionless group, the Nusselt number -- the ratio of convective to
conductive transport across the boundary layer:

    Nu = h L / k        =>        h = Nu k / L,

for a length L and fluid conductivity k. Nu itself follows correlations of the Reynolds and
Prandtl numbers. For turbulent flow in a pipe the workhorse is Dittus-Boelter,

    Nu = 0.023 Re^0.8 Pr^n         (n = 0.4 heating, 0.3 cooling),

and for a laminar boundary layer on a flat plate, Nu = 0.664 Re^0.5 Pr^(1/3). A lumped
object cooling in a draft loses temperature exponentially with a time constant
tau = rho c_p V / (h A), and whether the "lumped" assumption even holds is decided by the
Biot number Bi = h L / k_solid: if Bi << 1 the object stays nearly isothermal inside and only
its surface resistance matters.

This module gives Newton's cooling flux, the heat-transfer coefficient from Nu, the
Dittus-Boelter and flat-plate Nusselt correlations, the Biot number and the lumped cooling
law, and reproduces the ~10 W/(m^2 K) of gentle air and the hundreds-to-thousands of forced
water. SI units. Pure stdlib; the heat-transfer companion to the Peclet, Reynolds and
Rayleigh-Benard notes.
"""

from __future__ import annotations

import math


def newton_cooling_flux(h: float, t_surface: float, t_fluid: float) -> float:
    """Newton's law of cooling: convective heat flux q = h (T_s - T_inf) (W/m^2). Positive
    when the surface is hotter than the fluid (surface loses heat)."""
    return h * (t_surface - t_fluid)


def heat_transfer_coefficient(nusselt: float, conductivity: float, length: float) -> float:
    """h = Nu k / L (W/(m^2 K)) from the Nusselt number, the fluid conductivity k, and the
    characteristic length L."""
    return nusselt * conductivity / length


def dittus_boelter(reynolds: float, prandtl: float, heating: bool = True) -> float:
    """Dittus-Boelter Nusselt number for turbulent pipe flow: Nu = 0.023 Re^0.8 Pr^n, with
    n = 0.4 when the fluid is being heated, 0.3 when cooled. Valid for Re > ~1e4."""
    n = 0.4 if heating else 0.3
    return 0.023 * reynolds ** 0.8 * prandtl ** n


def flat_plate_laminar(reynolds: float, prandtl: float) -> float:
    """Average Nusselt number for a laminar boundary layer on a flat plate:
    Nu = 0.664 Re^0.5 Pr^(1/3). Valid for Re < ~5e5."""
    return 0.664 * math.sqrt(reynolds) * prandtl ** (1.0 / 3.0)


def biot_number(h: float, length: float, solid_conductivity: float) -> float:
    """Biot number Bi = h L / k_solid: surface convective resistance vs internal conductive
    resistance. Bi << 1 (~<0.1) means the body stays nearly isothermal inside -- the lumped-
    capacitance model applies."""
    return h * length / solid_conductivity


def lumped_time_constant(density: float, specific_heat: float, volume: float,
                         h: float, area: float) -> float:
    """Time constant tau = rho c_p V / (h A) (s) of a lumped object cooling by convection:
    its temperature excess decays as exp(-t/tau)."""
    return density * specific_heat * volume / (h * area)


def lumped_temperature(t: float, t_initial: float, t_fluid: float, tau: float) -> float:
    """Temperature at time t of a lumped object cooling from t_initial toward the fluid
    temperature: T(t) = T_inf + (T_0 - T_inf) exp(-t/tau)."""
    return t_fluid + (t_initial - t_fluid) * math.exp(-t / tau)
