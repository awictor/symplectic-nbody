"""Jeans escape: which gases a world can keep, and which leak to space.

At the top of an atmosphere (the exobase) gas molecules follow a Maxwell-Boltzmann
speed distribution set by the temperature T. A molecule moving upward faster than
the escape speed v_esc leaves for good. Light, hot molecules have a fatter high-speed
tail, so hydrogen boils off while nitrogen stays -- which is why Earth kept its N2
and CO2 but lost its primordial H2 and He, and why the hot low-gravity Moon has no
air at all.

The controlling quantity is the Jeans escape parameter, the ratio of gravitational
binding energy to thermal energy for one molecule at the exobase:

    lambda = v_esc^2 / v_th^2 = G M m / (R k_B T),   v_th = sqrt(2 k_B T / m).

A large lambda means escape is exponentially rare; the thermal (Jeans) escape flux
carries an exp(-lambda) Boltzmann factor. A rough rule of thumb: a gas is retained
over the age of the solar system when v_esc >~ 6 v_th (lambda >~ 36) and is lost
quickly when v_esc <~ a few v_th. The Jeans flux (particles per area per time) is

    Phi = n v_th (1 + lambda) exp(-lambda) / (2 sqrt(pi)),

the Maxwell-Boltzmann upward flux above v_esc.

This module gives the thermal speed, the escape parameter, the retention verdict,
and the Jeans flux, and reproduces the Earth-keeps-N2 / loses-H2 and airless-Moon
contrasts. SI units. Pure stdlib; the escape counterpart to the hydrostatic
atmosphere module.
"""

from __future__ import annotations

import math

G = 6.67430e-11
K_B = 1.380649e-23
AMU = 1.66053907e-27          # atomic mass unit (kg)

M_EARTH = 5.972e24
R_EARTH = 6.371e6
M_MOON = 7.342e22
R_MOON = 1.7374e6

# common molecular masses (kg)
M_H2 = 2.016 * AMU
M_HE = 4.0026 * AMU
M_H2O = 18.015 * AMU
M_N2 = 28.014 * AMU
M_O2 = 31.998 * AMU
M_CO2 = 44.01 * AMU

# rule-of-thumb: a gas is retained over ~Gyr when v_esc >= this many thermal speeds
RETENTION_RATIO = 6.0


def escape_speed(M: float, R: float) -> float:
    """Escape speed v_esc = sqrt(2 G M / R) (m/s) at radius R from a body of mass M."""
    return math.sqrt(2.0 * G * M / R)


def thermal_speed(T: float, m: float) -> float:
    """Most-probable thermal speed v_th = sqrt(2 k_B T / m) (m/s)."""
    return math.sqrt(2.0 * K_B * T / m)


def escape_parameter(M: float, R: float, T: float, m: float) -> float:
    """Jeans escape parameter lambda = v_esc^2 / v_th^2 = G M m / (R k_B T).
    Large lambda -> tightly bound -> escape exponentially suppressed."""
    return G * M * m / (R * K_B * T)


def is_retained(M: float, R: float, T: float, m: float,
                ratio: float = RETENTION_RATIO) -> bool:
    """True if the gas is retained over geological time: v_esc >= ratio * v_th,
    i.e. lambda >= ratio^2."""
    return escape_parameter(M, R, T, m) >= ratio * ratio


def jeans_flux(n: float, T: float, m: float, M: float, R: float) -> float:
    """Jeans thermal escape flux (particles m^-2 s^-1):
    Phi = n v_th (1 + lambda) exp(-lambda) / (2 sqrt(pi))."""
    v_th = thermal_speed(T, m)
    lam = escape_parameter(M, R, T, m)
    return n * v_th * (1.0 + lam) * math.exp(-lam) / (2.0 * math.sqrt(math.pi))
