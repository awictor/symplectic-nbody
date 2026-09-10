"""The Sackur-Tetrode equation: the absolute entropy of an ideal gas.

Classical thermodynamics gives only entropy differences. Quantum mechanics fixes the
constant: counting the distinguishable microstates of N indistinguishable atoms in a
box, with phase-space cells of size h^3, yields the Sackur-Tetrode equation for the
entropy of a monatomic ideal gas,

    S = N k_B [ ln( (V/N) (4 pi m U / (3 N h^2))^(3/2) ) + 5/2 ],

or, per particle and using U = (3/2) N k_B T,

    S/N k_B = ln( (k_B T / P) (2 pi m k_B T / h^2)^(3/2) ) + 5/2.

The quantum of action h enters explicitly -- without it the argument of the logarithm
would be dimensional and the entropy undefined. The famous 1/N! that makes entropy
extensive (resolving the Gibbs paradox that mixing identical gases should produce no
entropy) is built in.

Evaluated for argon at standard temperature and pressure it gives ~155 J/(mol K),
matching the measured standard molar entropy to within experimental error -- a direct
confirmation that entropy is countable microstates and that Planck's constant sets the
size of a phase-space cell.

This module gives the entropy per particle and total, the thermal de Broglie
wavelength, and the quantum-concentration check for when a gas stays classical, and
reproduces argon's standard molar entropy. SI units. Pure stdlib; the statistical-
mechanics companion to the Saha and degeneracy modules.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
H = 6.62607015e-34
N_A = 6.02214076e23
AMU = 1.66053907e-27


def thermal_wavelength(T: float, m: float) -> float:
    """Thermal de Broglie wavelength lambda = h / sqrt(2 pi m k_B T) (m). The gas is
    classical when the interparticle spacing greatly exceeds this."""
    return H / math.sqrt(2.0 * math.pi * m * K_B * T)


def quantum_concentration(T: float, m: float) -> float:
    """Quantum concentration n_Q = 1 / lambda^3 (m^-3). A gas is non-degenerate
    (classical) when its number density n << n_Q."""
    return 1.0 / thermal_wavelength(T, m) ** 3


def entropy_per_particle(T: float, P: float, m: float) -> float:
    """Sackur-Tetrode entropy per particle divided by k_B (dimensionless):
    S/(N k_B) = ln( (k_B T / P) (2 pi m k_B T / h^2)^(3/2) ) + 5/2."""
    term = (K_B * T / P) * (2.0 * math.pi * m * K_B * T / H ** 2) ** 1.5
    return math.log(term) + 2.5


def molar_entropy(T: float, P: float, m: float) -> float:
    """Standard molar entropy S (J/(mol K)) = N_A k_B * (S/N k_B)."""
    return N_A * K_B * entropy_per_particle(T, P, m)


def entropy_total(N: float, T: float, P: float, m: float) -> float:
    """Total entropy of N atoms (J/K)."""
    return N * K_B * entropy_per_particle(T, P, m)


def is_classical(n: float, T: float, m: float) -> bool:
    """True if the gas is classical (non-degenerate): number density n << n_Q,
    equivalently n lambda^3 << 1."""
    return n < 0.1 * quantum_concentration(T, m)
