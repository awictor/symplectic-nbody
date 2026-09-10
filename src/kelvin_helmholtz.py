"""The Kelvin-Helmholtz timescale: gravity as a (failed) stellar power source.

Before nuclear fusion was known, Kelvin and Helmholtz proposed the Sun shines by
slowly contracting, converting gravitational potential energy into heat and
light. The timescale to radiate away the star's gravitational binding energy is

    t_KH = G M^2 / (R L),

the Kelvin-Helmholtz (thermal) time. For the Sun this is only ~30 million years
-- famously far SHORT of the geological and biological evidence for a
billions-of-years-old Earth, which is how we know the Sun cannot be
gravity-powered: it needs nuclear fusion. But t_KH is exactly the timescale on
which a PRE-main-sequence protostar contracts before fusion ignites, and the
timescale on which a star relaxes if its thermal balance is disturbed.

This module gives the KH time, contrasts it with the nuclear (main-sequence)
lifetime, and reproduces the ~30 Myr solar value. SI units and solar units.
Pure stdlib; complements main_sequence and virial.
"""

from __future__ import annotations

import math

G = 6.67430e-11
M_SUN = 1.98892e30
R_SUN = 6.957e8
L_SUN = 3.828e26
YEAR = 3.15576e7


def kelvin_helmholtz_time(M: float, R: float, L: float) -> float:
    """Thermal (Kelvin-Helmholtz) time t = G M^2 / (R L), in seconds."""
    return G * M * M / (R * L)


def solar_kh_time_myr() -> float:
    """The Sun's Kelvin-Helmholtz time in Myr (~30)."""
    return kelvin_helmholtz_time(M_SUN, R_SUN, L_SUN) / YEAR / 1e6


def kh_time_solar_units(mass_msun: float, radius_rsun: float,
                        lum_lsun: float) -> float:
    """KH time in Myr for a star in solar units: scales as M^2 / (R L)."""
    M = mass_msun * M_SUN
    R = radius_rsun * R_SUN
    L = lum_lsun * L_SUN
    return kelvin_helmholtz_time(M, R, L) / YEAR / 1e6


def gravitational_binding_energy(M: float, R: float) -> float:
    """Order-of-magnitude gravitational binding energy ~ G M^2 / R (J).
    (The full number carries a structure factor of order unity, e.g. 3/5 for a
    uniform sphere; t_KH uses this energy divided by the luminosity.)"""
    return G * M * M / R
