"""The van der Waals gas: a real gas that can condense.

The ideal gas law ignores two things real molecules have: a finite size, and mutual
attraction. Van der Waals patched both into the equation of state,

    (P + a n^2 / V^2)(V - n b) = n R T,

where a measures the attraction (it lowers the pressure the gas exerts) and b the volume
the molecules themselves occupy (it shrinks the available volume). Small as they are,
these corrections give the gas something the ideal law never can: a liquid-vapour phase
transition.

Below a critical temperature the P-V isotherm develops a wiggle -- a region where
pressure would rise with volume, which is unstable and where the gas condenses. At the
critical point that wiggle shrinks to an inflection, fixing the critical constants in
terms of a and b alone:

    T_c = 8a / (27 R b),   V_c = 3 n b,   P_c = a / (27 b^2),

and the dimensionless compressibility there is universal, P_c V_c / (n R T_c) = 3/8 for
every van der Waals gas. Written in reduced variables (P/P_c, V/V_c, T/T_c) all gases
collapse onto one law of corresponding states.

This module gives the vdW pressure, the critical constants, the compressibility factor,
and the reduced-variable form, and reproduces CO2's ~304 K critical temperature and the
universal 3/8 ratio. SI units. Pure stdlib; the real-gas companion to the Sackur-Tetrode
and adiabatic modules.
"""

from __future__ import annotations

R_GAS = 8.314462618

# van der Waals constants (SI: a in Pa m^6/mol^2, b in m^3/mol)
A_CO2, B_CO2 = 0.3640, 4.267e-5
A_WATER, B_WATER = 0.5536, 3.049e-5
A_HELIUM, B_HELIUM = 0.00346, 2.38e-5


def pressure(n: float, V: float, T: float, a: float, b: float) -> float:
    """van der Waals pressure P = nRT/(V - nb) - a n^2/V^2 (Pa)."""
    return n * R_GAS * T / (V - n * b) - a * n * n / (V * V)


def critical_temperature(a: float, b: float) -> float:
    """Critical temperature T_c = 8a / (27 R b) (K)."""
    return 8.0 * a / (27.0 * R_GAS * b)


def critical_volume(b: float, n: float = 1.0) -> float:
    """Critical (molar) volume V_c = 3 n b (m^3)."""
    return 3.0 * n * b


def critical_pressure(a: float, b: float) -> float:
    """Critical pressure P_c = a / (27 b^2) (Pa)."""
    return a / (27.0 * b * b)


def critical_compressibility(a: float, b: float) -> float:
    """Compressibility at the critical point P_c V_c / (R T_c) = 3/8 -- universal for
    every van der Waals gas (a, b cancel)."""
    Tc = critical_temperature(a, b)
    Pc = critical_pressure(a, b)
    Vc = critical_volume(b, 1.0)
    return Pc * Vc / (R_GAS * Tc)


def compressibility_factor(n: float, V: float, T: float, a: float, b: float) -> float:
    """Compressibility Z = PV/(nRT): Z=1 for an ideal gas, <1 when attraction dominates,
    >1 when finite size dominates."""
    P = pressure(n, V, T, a, b)
    return P * V / (n * R_GAS * T)


def reduced_pressure(Pr: float, Vr: float, Tr: float) -> float:
    """The law of corresponding states in reduced variables (P/P_c, V/V_c, T/T_c):
    (Pr + 3/Vr^2)(3 Vr - 1) = 8 Tr. Returns the residual (0 when consistent) given Pr,
    Vr, Tr -- or solve for Pr: Pr = 8 Tr/(3 Vr - 1) - 3/Vr^2."""
    return 8.0 * Tr / (3.0 * Vr - 1.0) - 3.0 / (Vr * Vr)
