"""Equipartition: half a k_B T per degree of freedom, and the heat capacity of gases.

Classical statistical mechanics gives a strikingly simple rule: every quadratic degree of
freedom in a system's energy -- each translational velocity component, each rotational axis,
each vibrational coordinate (which counts twice, kinetic + potential) -- carries a mean
energy of (1/2) k_B T in thermal equilibrium. Sum them and you get the internal energy, and
its temperature derivative is the heat capacity.

For an ideal gas with f active degrees of freedom per molecule:

    U   = (f/2) n R T
    C_V = (f/2) R          C_P = C_V + R          gamma = C_P/C_V = (f+2)/f.

So a monatomic gas (f=3, translation only) has C_V = 3R/2 and gamma = 5/3; a diatomic gas
at room temperature (f=5, + two rotations) has C_V = 5R/2 and gamma = 7/5; adding the
vibrational mode (f=7) gives C_V = 7R/2. A solid, with three vibrational modes per atom
counting kinetic and potential, reaches f=6 and C_V = 3R -- the Dulong-Petit law.

Equipartition is the CLASSICAL, high-temperature limit. Quantum mechanics freezes a mode out
once k_B T drops below its energy quantum, so a real gas climbs a heat-capacity staircase as
it warms: translation (always on) -> rotation (above ~tens of K) -> vibration (above ~1000s
of K). This module gives the per-mode energy, the internal energy and molar C_V/C_P/gamma
for a chosen f, the rms thermal speed, and a simple two-level (Schottky-style) activation
factor for a mode, and reproduces the monatomic and diatomic heat capacities and the H2
staircase. SI units. Pure stdlib; the kinetic-theory companion to the Maxwell-Boltzmann,
Sackur-Tetrode and Debye notes.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23              # Boltzmann constant (J/K)
R_GAS = 8.314462618            # molar gas constant (J/(mol K))
N_A = 6.02214076e23


def energy_per_dof(temperature: float) -> float:
    """Mean thermal energy of one quadratic degree of freedom: (1/2) k_B T (J)."""
    return 0.5 * K_B * temperature


def internal_energy(dof: float, temperature: float, moles: float = 1.0) -> float:
    """Internal energy U = (f/2) n R T (J) for f active degrees of freedom."""
    return 0.5 * dof * moles * R_GAS * temperature


def molar_cv(dof: float) -> float:
    """Molar heat capacity at constant volume C_V = (f/2) R (J/(mol K))."""
    return 0.5 * dof * R_GAS


def molar_cp(dof: float) -> float:
    """Molar heat capacity at constant pressure C_P = C_V + R (J/(mol K)), Mayer's relation."""
    return molar_cv(dof) + R_GAS


def gamma_ratio(dof: float) -> float:
    """Adiabatic index gamma = C_P/C_V = (f + 2)/f."""
    return (dof + 2.0) / dof


def rms_speed(temperature: float, molar_mass: float) -> float:
    """Root-mean-square molecular speed sqrt(3 R T / M) (m/s) -- the three translational
    degrees of freedom, each (1/2)k_B T, give (1/2)m<v^2> = (3/2)k_B T."""
    return math.sqrt(3.0 * R_GAS * temperature / molar_mass)


def mode_activation(temperature: float, characteristic_temperature: float) -> float:
    """Fraction (0..1) of a mode's full classical heat-capacity contribution that is active,
    using the Einstein two-state heat-capacity factor

        (theta/T)^2 e^(theta/T) / (e^(theta/T) - 1)^2,

    which -> 1 for T >> theta (mode fully classical) and -> 0 for T << theta (frozen out).
    characteristic_temperature theta is the rotational or vibrational temperature of the mode."""
    if temperature <= 0.0:
        return 0.0
    x = characteristic_temperature / temperature
    if x > 350.0:               # exp(x)^2 would overflow float; mode is deeply frozen
        return 0.0
    ex = math.exp(x)
    return x * x * ex / (ex - 1.0) ** 2


def effective_cv_diatomic(temperature: float, theta_rot: float, theta_vib: float) -> float:
    """Molar C_V (J/(mol K)) of a diatomic gas including quantum freeze-out: 3 translational
    half-R's (always on) + 2 rotational (activated above theta_rot) + 2 vibrational
    (activated above theta_vib, kinetic+potential). Climbs 3R/2 -> 5R/2 -> 7R/2 with T."""
    trans = 1.5 * R_GAS
    rot = R_GAS * mode_activation(temperature, theta_rot)          # 2 half-R modes = R
    vib = R_GAS * mode_activation(temperature, theta_vib)          # kinetic + potential = R
    return trans + rot + vib
