"""Adiabatic processes: compression and expansion without heat exchange.

When a gas is compressed or expanded fast enough that no heat flows (adiabatic), it
obeys the Poisson relations

    P V^gamma = const,      T V^(gamma-1) = const,      T P^((1-gamma)/gamma) = const,

with gamma = C_p/C_v the adiabatic index (5/3 monatomic, 7/5 diatomic). Unlike an
isothermal process the temperature changes: compressing a gas heats it (a diesel engine
ignites fuel by adiabatic compression alone), expanding it cools it (why released spray
is cold, and why rising air cools at the dry adiabatic lapse rate).

The work done in an adiabatic expansion is W = (P1 V1 - P2 V2)/(gamma - 1), and the
associated speed of sound is c = sqrt(gamma P/rho) = sqrt(gamma R T/M) -- adiabatic, not
isothermal, because sound oscillations are too fast to exchange heat (Laplace's
correction to Newton, which fixed the ~20% error in the predicted speed of sound).

This module gives the pressure/temperature/volume relations, the adiabatic work, and
the adiabatic sound speed, and reproduces diesel-compression heating and the corrected
speed of sound in air. SI units. Pure stdlib; the thermodynamic-process companion to
the Carnot and Brunt-Vaisala modules.
"""

from __future__ import annotations

import math

R_GAS = 8.314462618
GAMMA_MONO = 5.0 / 3.0
GAMMA_DIATOMIC = 7.0 / 5.0


def pressure_after(P1: float, V1: float, V2: float,
                   gamma: float = GAMMA_DIATOMIC) -> float:
    """Pressure after adiabatic volume change: P2 = P1 (V1/V2)^gamma."""
    return P1 * (V1 / V2) ** gamma


def temperature_after_volume(T1: float, V1: float, V2: float,
                             gamma: float = GAMMA_DIATOMIC) -> float:
    """Temperature after adiabatic volume change: T2 = T1 (V1/V2)^(gamma-1).
    Compression (V2<V1) heats; expansion cools."""
    return T1 * (V1 / V2) ** (gamma - 1.0)


def temperature_after_pressure(T1: float, P1: float, P2: float,
                               gamma: float = GAMMA_DIATOMIC) -> float:
    """Temperature after adiabatic pressure change: T2 = T1 (P2/P1)^((gamma-1)/gamma)."""
    return T1 * (P2 / P1) ** ((gamma - 1.0) / gamma)


def adiabatic_work(P1: float, V1: float, P2: float, V2: float,
                   gamma: float = GAMMA_DIATOMIC) -> float:
    """Work done BY the gas in an adiabatic process: W = (P1 V1 - P2 V2)/(gamma-1).
    Positive for expansion (gas does work and cools)."""
    return (P1 * V1 - P2 * V2) / (gamma - 1.0)


def sound_speed(T: float, molar_mass: float, gamma: float = GAMMA_DIATOMIC) -> float:
    """Adiabatic speed of sound c = sqrt(gamma R T / M) (m/s), M the molar mass (kg/mol).
    Laplace's gamma factor corrects Newton's isothermal estimate."""
    return math.sqrt(gamma * R_GAS * T / molar_mass)


def compression_ratio_for_temperature(T1: float, T2: float,
                                      gamma: float = GAMMA_DIATOMIC) -> float:
    """Volume compression ratio V1/V2 needed to reach temperature T2 from T1
    adiabatically: (T2/T1)^(1/(gamma-1))."""
    return (T2 / T1) ** (1.0 / (gamma - 1.0))
