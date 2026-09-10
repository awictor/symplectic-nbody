"""Atmospheric escape: why Earth keeps nitrogen but loses hydrogen.

A planet holds onto a gas only if its gravity beats the gas molecules' thermal
motion. The comparison is the Jeans escape parameter

    lambda = v_esc^2 / v_th^2 = G M m / (r k T),

the ratio of gravitational binding energy to thermal energy for a molecule of
mass m. Large lambda -> the Maxwell tail above escape speed is negligible and the
gas is retained; small lambda -> the atmosphere boils off.

The rule of thumb: a species is retained over the age of the Solar System if the
escape speed exceeds ~6 times the most-probable thermal speed (lambda >~ 36 gives
a Jeans lifetime longer than ~Gyr). That is why Earth (v_esc = 11.2 km/s) keeps
N2 and O2 but slowly loses H2 and He, the Moon (v_esc = 2.4 km/s) keeps nothing,
and giant planets keep even hydrogen.

This module gives the thermal speed, the escape parameter, and a retention
verdict, and reproduces those cases. SI units. Pure stdlib; complements
cosmic_velocities.
"""

from __future__ import annotations

import math

G = 6.67430e-11
K_B = 1.380649e-23
M_U = 1.66053907e-27           # atomic mass unit, kg


def thermal_speed(T: float, molar_mass_amu: float) -> float:
    """Most-probable thermal speed v_th = sqrt(2 k T / m) (m/s)."""
    m = molar_mass_amu * M_U
    return math.sqrt(2.0 * K_B * T / m)


def escape_speed(M_planet: float, r: float) -> float:
    """Escape speed sqrt(2 G M / r) (m/s)."""
    return math.sqrt(2.0 * G * M_planet / r)


def escape_parameter(M_planet: float, r: float, T: float,
                     molar_mass_amu: float) -> float:
    """Jeans escape parameter lambda = v_esc^2 / v_th^2. Large -> retained."""
    ve = escape_speed(M_planet, r)
    vth = thermal_speed(T, molar_mass_amu)
    return (ve * ve) / (vth * vth)


def is_retained(M_planet: float, r: float, T: float, molar_mass_amu: float,
                threshold: float = 36.0) -> bool:
    """A gas is retained over Solar-System timescales if lambda exceeds ~36
    (escape speed >~ 6 x thermal speed)."""
    return escape_parameter(M_planet, r, T, molar_mass_amu) >= threshold
