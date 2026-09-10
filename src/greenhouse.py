"""The greenhouse effect: why planets are warmer than sunlight alone allows.

A planet in radiative balance with its star settles at the equilibrium temperature

    T_eq = T_sun sqrt(R_sun / 2 d) (1 - A)^(1/4),

the blackbody temperature that re-radiates the absorbed sunlight. But an atmosphere
opaque in the infrared changes the game: it lets visible sunlight in but absorbs the
planet's outgoing infrared and re-radiates half of it back down. The surface must then
run hotter to push the same energy out through the blanket.

For a grey atmosphere of infrared optical depth tau, the surface temperature is

    T_surf = T_eq (1 + 3 tau / 4)^(1/4),

so the surface warms above the equilibrium (skin) temperature by a factor that grows
with optical depth. Earth's tau ~ 0.6 lifts its 255 K equilibrium temperature to the
observed ~288 K -- a 33 K greenhouse warming that keeps the oceans liquid. Venus, with
a massive CO2 atmosphere of tau ~ 120, is heated from a ~230 K equilibrium temperature
to a surface-melting ~737 K -- the runaway greenhouse.

This module gives the equilibrium temperature, the grey-atmosphere surface temperature,
the greenhouse warming, and the optical depth needed to reach a target surface
temperature, and reproduces Earth's +33 K and Venus's runaway. SI-friendly (solar
units for convenience). Pure stdlib; the atmospheric-warming companion to the
blackbody and habitable-zone modules.
"""

from __future__ import annotations

import math

# zero-albedo equilibrium temperature at 1 AU from the Sun (K)
T_EQ_1AU = 278.5


def equilibrium_temperature(L_lsun: float, d_au: float,
                            albedo: float = 0.3) -> float:
    """Planet equilibrium (skin) temperature (K): T_eq = 278.5 (1-A)^(1/4)
    L^(1/4) / sqrt(d), with L in solar luminosities and d in AU."""
    return T_EQ_1AU * (1.0 - albedo) ** 0.25 * L_lsun ** 0.25 / math.sqrt(d_au)


def surface_temperature(T_eq: float, tau: float) -> float:
    """Grey-atmosphere surface temperature (K): T_surf = T_eq (1 + 3 tau / 4)^(1/4).
    tau is the infrared optical depth; tau=0 gives the airless equilibrium value."""
    return T_eq * (1.0 + 0.75 * tau) ** 0.25


def greenhouse_warming(T_eq: float, tau: float) -> float:
    """Surface warming (K) due to the greenhouse blanket: T_surf - T_eq."""
    return surface_temperature(T_eq, tau) - T_eq


def optical_depth_for_surface(T_eq: float, T_surf: float) -> float:
    """Infrared optical depth needed to warm a surface from T_eq to T_surf:
    tau = (4/3)((T_surf/T_eq)^4 - 1)."""
    return (4.0 / 3.0) * ((T_surf / T_eq) ** 4 - 1.0)


def surface_temperature_planet(L_lsun: float, d_au: float, albedo: float,
                               tau: float) -> float:
    """Convenience: surface temperature straight from orbit, albedo and optical depth."""
    return surface_temperature(equilibrium_temperature(L_lsun, d_au, albedo), tau)
