"""The Clausius-Clapeyron relation: how vapor pressure climbs with temperature.

Along a liquid-vapour coexistence line, pressure and temperature are locked together by

    dP/dT = L / (T dV),

with L the latent heat of vaporization and dV the volume change on boiling. Treating the
vapour as an ideal gas and neglecting the liquid volume, this integrates to

    P(T) = P0 exp( -(L/R) (1/T - 1/T0) ),

a steep exponential: vapor pressure roughly doubles every ~10-15 K near room temperature.
Turned around, the boiling point is where the vapor pressure equals the ambient
pressure, so water boils below 100 C on a mountain (lower air pressure) and a pressure
cooker raises the boiling point to cook faster.

The same relation gives the latent heat from two measured (P, T) points, describes the
melting curve (with the latent heat of fusion), and underlies the saturation vapor
pressure that sets atmospheric humidity and cloud formation.

This module gives the vapor pressure at a temperature, the boiling point at a given
ambient pressure, the latent heat from two data points, and the fractional pressure
change per degree, and reproduces water's 100 C boiling at 1 atm and its drop with
altitude. SI units, temperatures in K. Pure stdlib; the phase-transition companion to
the van-der-Waals and Saha modules.
"""

from __future__ import annotations

import math

R_GAS = 8.314462618

# water reference: latent heat of vaporization (J/mol), boils at 373.15 K at 1 atm
L_WATER = 40660.0
T_BOIL_WATER = 373.15
P_ATM = 101325.0


def vapor_pressure(T: float, L: float = L_WATER, T0: float = T_BOIL_WATER,
                   P0: float = P_ATM) -> float:
    """Saturation vapor pressure (Pa) at temperature T from Clausius-Clapeyron:
    P = P0 exp(-(L/R)(1/T - 1/T0))."""
    return P0 * math.exp(-(L / R_GAS) * (1.0 / T - 1.0 / T0))


def boiling_point(P_ambient: float, L: float = L_WATER, T0: float = T_BOIL_WATER,
                  P0: float = P_ATM) -> float:
    """Boiling temperature (K) at ambient pressure P_ambient: invert the vapor-pressure
    curve, 1/T = 1/T0 - (R/L) ln(P/P0)."""
    return 1.0 / (1.0 / T0 - (R_GAS / L) * math.log(P_ambient / P0))


def latent_heat_from_two_points(P1: float, T1: float, P2: float, T2: float) -> float:
    """Latent heat (J/mol) from two coexistence points: L = R ln(P2/P1) / (1/T1 - 1/T2)."""
    return R_GAS * math.log(P2 / P1) / (1.0 / T1 - 1.0 / T2)


def fractional_pressure_change(T: float, L: float = L_WATER) -> float:
    """Fractional change in vapor pressure per kelvin, (1/P) dP/dT = L/(R T^2)."""
    return L / (R_GAS * T * T)


def pressure_from_altitude(altitude_m: float, P0: float = P_ATM,
                           scale_height: float = 8400.0) -> float:
    """Ambient pressure (Pa) at an altitude using the isothermal barometric formula
    P = P0 exp(-h/H), for converting altitude to a boiling point."""
    return P0 * math.exp(-altitude_m / scale_height)
