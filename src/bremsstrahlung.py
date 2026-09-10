"""Bremsstrahlung: the free-free X-rays of hot ionized gas.

When a free electron is deflected by an ion's Coulomb field it accelerates and
radiates (Larmor) -- "braking radiation," or bremsstrahlung. For a hot, thin,
fully ionized plasma the thermal free-free volume emissivity is

    epsilon_ff ~ 1.4e-40 Z^2 n_e n_i sqrt(T)   W/m^3   (SI, cgs-derived constant)

-- proportional to the square of the density (two-body process) and the square
root of temperature. This is how galaxy-cluster gas at ~10^7-10^8 K shines in
X-rays (the emission whose CMB imprint is the SZ effect).

The gas COOLING TIME is the thermal energy divided by the emission rate,

    t_cool ~ (3/2) n k T / epsilon_ff  ~  sqrt(T) / n,

so dense cluster cores cool in less than a Hubble time (cooling flows) while the
tenuous outskirts effectively never cool. This module gives the emissivity, the
cooling time, and their scalings, and reproduces the cluster regime. SI units.
Pure stdlib; the free-free counterpart to synchrotron, rooted in Larmor.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
GYR = 3.156e16
C_FF = 1.4e-40                 # thermal bremsstrahlung constant (SI, W m^3 K^-1/2)


def emissivity(n_e: float, T: float, Z: float = 1.0, n_i: float = None) -> float:
    """Thermal free-free volume emissivity (W/m^3):
    epsilon = C_ff Z^2 n_e n_i sqrt(T). Defaults n_i = n_e (hydrogen plasma)."""
    if n_i is None:
        n_i = n_e
    return C_FF * Z * Z * n_e * n_i * math.sqrt(T)


def cooling_time(n_e: float, T: float, Z: float = 1.0) -> float:
    """Radiative cooling time t = (3 n k T) / epsilon (s), taking the thermal
    energy density as ~3 n_e k T (electrons + ions)."""
    eps = emissivity(n_e, T, Z)
    return 3.0 * n_e * K_B * T / eps


def cooling_time_gyr(n_e: float, T: float, Z: float = 1.0) -> float:
    """Cooling time in Gyr."""
    return cooling_time(n_e, T, Z) / GYR


def cools_within_hubble(n_e: float, T: float, Z: float = 1.0,
                        hubble_gyr: float = 13.8) -> bool:
    """True if the gas can radiate away its thermal energy within a Hubble time
    (a cooling-flow candidate)."""
    return cooling_time_gyr(n_e, T, Z) < hubble_gyr
