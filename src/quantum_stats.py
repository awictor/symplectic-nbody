"""Quantum statistics: how identical particles share energy states.

Identical quantum particles come in two kinds, and their statistics decide everything
from why metals conduct to why lasers work. The average occupation of a state at energy
E, temperature T, and chemical potential mu is

    Fermi-Dirac    <n> = 1 / (exp((E - mu)/kT) + 1)      (fermions: electrons, protons)
    Bose-Einstein  <n> = 1 / (exp((E - mu)/kT) - 1)      (bosons: photons, He-4)
    Maxwell-Boltzmann <n> = exp(-(E - mu)/kT)            (classical limit)

The tiny +1 / -1 makes all the difference. FERMIONS obey the Pauli exclusion principle:
occupation never exceeds 1, so at T=0 they fill every state up to the Fermi energy mu in
a sharp step (electron degeneracy, white-dwarf pressure). BOSONS have no such limit; as
E -> mu the occupation diverges, and below a critical temperature a macroscopic fraction
piles into the ground state -- Bose-Einstein condensation. Both reduce to the classical
Maxwell-Boltzmann exponential when the states are sparsely occupied (E - mu >> kT), the
regime of an ordinary gas.

This module gives the three occupation numbers, the Fermi-Dirac step width, and the
photon-gas occupation (mu=0), and reproduces the sharp T=0 Fermi step, the bosonic
divergence, and the classical limit. SI units, energies in eV via a helper. Pure
stdlib; the statistical-mechanics companion to the degeneracy and Sackur-Tetrode modules.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
EV = 1.602176634e-19


def fermi_dirac(E: float, mu: float, T: float) -> float:
    """Fermi-Dirac occupation <n> = 1/(exp((E-mu)/kT)+1). In [0,1] (Pauli exclusion)."""
    if T <= 0.0:
        return 1.0 if E < mu else (0.5 if E == mu else 0.0)
    x = (E - mu) / (K_B * T)
    if x > 700.0:
        return 0.0
    return 1.0 / (math.exp(x) + 1.0)


def bose_einstein(E: float, mu: float, T: float) -> float:
    """Bose-Einstein occupation <n> = 1/(exp((E-mu)/kT)-1). Diverges as E -> mu;
    requires E > mu."""
    x = (E - mu) / (K_B * T)
    if x <= 0.0:
        return float("inf")
    if x > 700.0:
        return 0.0
    return 1.0 / (math.exp(x) - 1.0)


def maxwell_boltzmann(E: float, mu: float, T: float) -> float:
    """Classical Maxwell-Boltzmann occupation <n> = exp(-(E-mu)/kT): the sparse-state
    limit of both quantum distributions."""
    return math.exp(-(E - mu) / (K_B * T))


def photon_occupation(E: float, T: float) -> float:
    """Bose-Einstein occupation of a photon mode (mu = 0): the Planck factor
    1/(exp(E/kT)-1)."""
    return bose_einstein(E, 0.0, T)


def fermi_step_width(T: float) -> float:
    """Approximate energy width (J) over which the Fermi-Dirac step falls from ~0.9 to
    ~0.1: about 4 kT (zero at T=0, a sharp step)."""
    return 4.0 * K_B * T


def is_classical(E: float, mu: float, T: float, tol: float = 0.01) -> bool:
    """True if the state is sparsely occupied (<n> << 1), where quantum and classical
    statistics agree: occupation below `tol`."""
    return maxwell_boltzmann(E, mu, T) < tol
