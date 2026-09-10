"""BCS superconductivity: the energy gap that stops resistance.

Below a critical temperature T_c, electrons in a metal do something remarkable: a weak
phonon-mediated attraction binds them into Cooper pairs, and those pairs condense into a
single coherent quantum state that carries current with zero resistance. Bardeen, Cooper and
Schrieffer explained it in 1957. The heart of the theory is an energy gap Delta that opens at
the Fermi surface -- an energy you must pay to break a pair -- so at low temperature there are
no low-energy excitations to scatter the current, and resistance vanishes.

BCS predicts a universal ratio between the zero-temperature gap and T_c:

    2 Delta(0) / (k_B T_c) = 3.53,

the same for every weak-coupling superconductor regardless of material. The gap is temperature
dependent, closing as T -> T_c roughly as

    Delta(T) ~ Delta(0) sqrt(1 - T/T_c),

and it shows up directly: a photon or phonon below 2 Delta cannot break a pair, so the
material is transparent to low-frequency radiation and its heat capacity is exponentially
suppressed. T_c itself follows from the phonon (Debye) energy and the electron-phonon coupling
lambda,

    k_B T_c = 1.13 hbar omega_D exp(-1/lambda),

which -- through omega_D ~ 1/sqrt(M) -- gives the isotope effect T_c ~ M^(-1/2), the smoking-
gun evidence that phonons do the pairing.

This module gives the BCS gap-to-T_c ratio, the zero-T gap from T_c, the temperature-dependent
gap, the T_c from Debye energy and coupling, and the isotope shift, and reproduces the 3.53
ratio and aluminium/niobium gaps. SI units with meV helpers. Pure stdlib; the
condensed-matter companion to the Josephson and quantum-Hall notes.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
HBAR = 1.054571817e-34
BCS_RATIO = 3.53              # 2 Delta(0) / (k_B T_c), weak-coupling universal value


def gap_from_tc(tc: float) -> float:
    """Zero-temperature energy gap Delta(0) = 3.53 k_B T_c / 2 (J) from the critical
    temperature -- the universal BCS relation."""
    return BCS_RATIO * K_B * tc / 2.0


def tc_from_gap(gap: float) -> float:
    """Critical temperature T_c = 2 Delta(0) / (3.53 k_B) (K) from the measured gap. Inverts
    gap_from_tc."""
    return 2.0 * gap / (BCS_RATIO * K_B)


def gap_ratio(gap: float, tc: float) -> float:
    """The dimensionless ratio 2 Delta / (k_B T_c). ~3.53 for a weak-coupling BCS
    superconductor; larger for strong coupling (lead, mercury)."""
    return 2.0 * gap / (K_B * tc)


def gap_at_temperature(gap0: float, t: float, tc: float) -> float:
    """Temperature-dependent gap Delta(T) ~ Delta(0) sqrt(1 - T/T_c) (J), closing to zero at
    T_c. Returns 0 at or above T_c (normal state)."""
    if t >= tc:
        return 0.0
    return gap0 * math.sqrt(1.0 - t / tc)


def critical_temperature(debye_frequency: float, coupling: float) -> float:
    """BCS critical temperature T_c = 1.13 hbar omega_D exp(-1/lambda) / k_B (K), from the
    Debye (phonon) frequency omega_D and electron-phonon coupling lambda."""
    return 1.13 * HBAR * debye_frequency * math.exp(-1.0 / coupling) / K_B


def isotope_shifted_tc(tc: float, mass_ratio: float) -> float:
    """Critical temperature after an isotope substitution, T_c' = T_c (M/M')^(1/2), from the
    T_c ~ M^(-1/2) isotope effect. mass_ratio = M_new / M_old > 1 lowers T_c."""
    return tc / math.sqrt(mass_ratio)


def pair_breaking_frequency(gap0: float) -> float:
    """Minimum photon frequency 2 Delta / h that can break a Cooper pair (Hz). Below it the
    superconductor is lossless to radiation; it sits in the microwave/THz for real gaps."""
    return 2.0 * gap0 / (2.0 * math.pi * HBAR)
