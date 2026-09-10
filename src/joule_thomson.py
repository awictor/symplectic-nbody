"""The Joule-Thomson effect: cooling a gas by pushing it through a valve.

Force a real gas through a porous plug or narrow valve from high to low pressure at
constant enthalpy and its temperature changes -- the Joule-Thomson effect, the workhorse
of gas liquefaction and refrigeration. The sign is set by the JT coefficient

    mu_JT = (dT/dP)_H = (1/C_p)(T (dV/dT)_P - V),

which for a van der Waals gas is approximately mu_JT ~ (1/C_p)(2a/RT - b). Two competing
effects: molecular attraction (a) cools the gas as it expands (work against attraction),
while finite size (b) warms it. Which wins depends on temperature.

The crossover is the inversion temperature; below it expansion cools (mu_JT > 0), above
it warms (mu_JT < 0). The maximum inversion temperature is

    T_inv = 2a / (R b) = (27/4) T_c,

so most gases (nitrogen T_inv ~ 620 K, well above room temperature) cool when throttled
and can be liquefied by repeated expansion -- but hydrogen (T_inv ~ 200 K) and helium
(T_inv ~ 40 K) must first be pre-cooled below their inversion temperatures, or throttling
heats them. An ideal gas has mu_JT = 0 exactly (no attraction, no size).

This module gives the JT coefficient, the inversion temperature, the cooling/heating
verdict, the temperature change for a pressure drop, and the ratio to the critical
temperature, and reproduces nitrogen cooling and helium needing pre-cooling. SI units.
Pure stdlib; the real-gas-throttling companion to the van-der-Waals and adiabatic modules.
"""

from __future__ import annotations

R_GAS = 8.314462618

# van der Waals a (Pa m^6/mol^2), b (m^3/mol), and molar C_p (J/mol/K)
A_N2, B_N2, CP_N2 = 0.1370, 3.87e-5, 29.1
A_H2, B_H2, CP_H2 = 0.02476, 2.661e-5, 28.8
A_HE, B_HE, CP_HE = 0.00346, 2.38e-5, 20.8
A_CO2, B_CO2, CP_CO2 = 0.3640, 4.267e-5, 37.1


def jt_coefficient(T: float, a: float, b: float, C_p: float) -> float:
    """Joule-Thomson coefficient mu = (dT/dP)_H ~ (1/C_p)(2a/RT - b) (K/Pa) for a van
    der Waals gas. Positive -> throttling cools; negative -> warms."""
    return (2.0 * a / (R_GAS * T) - b) / C_p


def inversion_temperature(a: float, b: float) -> float:
    """Maximum inversion temperature T_inv = 2a / (R b) (K): above it the gas warms on
    throttling, below it cools."""
    return 2.0 * a / (R_GAS * b)


def cools_on_expansion(T: float, a: float, b: float) -> bool:
    """True if throttling cools the gas at temperature T (T below the inversion
    temperature, mu_JT > 0)."""
    return T < inversion_temperature(a, b)


def temperature_change(T: float, dP: float, a: float, b: float,
                       C_p: float) -> float:
    """Approximate temperature change (K) for a pressure DROP dP (>0, Pa) through the
    valve: dT = -mu_JT dP (negative dP through the valve, so dT = mu * (-dP))."""
    return -jt_coefficient(T, a, b, C_p) * dP


def inversion_over_critical(a: float, b: float) -> float:
    """Ratio of the maximum inversion temperature to the critical temperature:
    T_inv / T_c = 27/4 = 6.75 for a van der Waals gas."""
    Tc = 8.0 * a / (27.0 * R_GAS * b)
    return inversion_temperature(a, b) / Tc
