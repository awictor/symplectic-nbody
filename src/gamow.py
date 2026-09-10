"""The Gamow peak: the narrow energy window where stars fuse.

Two nuclei must overcome their mutual Coulomb repulsion to fuse, a barrier of order an
MeV -- yet the centre of the Sun is only ~1.3 keV (15 million K). Classically fusion
would be impossible. Two quantum/statistical effects rescue it, pulling in opposite
directions in energy:

  * the Maxwell-Boltzmann tail, exp(-E / k_B T), which supplies fewer particles as E
    rises;
  * quantum tunnelling through the Coulomb barrier, whose Gamow factor
    exp(-b / sqrt(E)) makes penetration far more likely as E rises,
    with b = pi Z1 Z2 e^2 sqrt(2 mu) / (2 pi eps0 hbar) ... (the Gamow energy is
    E_G = b^2).

Their product exp(-E/kT - b/sqrt(E)) is sharply peaked at the Gamow peak energy

    E0 = (b k_B T / 2)^(2/3) = (E_G (k_B T)^2 / 4)^(1/3),

far out on the thermal tail but well below the barrier top. Almost all fusion happens
in a narrow window around E0 -- for proton-proton fusion in the Sun, E0 ~ 6 keV, several
times the mean thermal energy. Because E0 and the peak height depend so steeply on
charge and temperature, fusion rates are ferociously sensitive to both: raising Z or
lowering T shifts the peak and can shut a reaction off entirely, which is why heavier
elements burn only in hotter cores.

This module gives the Gamow energy, the Gamow peak energy and width, and the relative
reaction integrand, and reproduces the ~6 keV solar p-p peak. SI units, energies in
joules with a keV helper. Pure stdlib; the nuclear-burning companion to the
Maxwell-Boltzmann and main-sequence modules.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
HBAR = 1.054571817e-34
M_P = 1.6726219e-27
KEV = 1e3 * E_CHARGE


def reduced_mass(m1: float, m2: float) -> float:
    """Reduced mass mu = m1 m2 / (m1 + m2) (kg)."""
    return m1 * m2 / (m1 + m2)


def gamow_energy(Z1: float, Z2: float, mu: float) -> float:
    """Gamow energy E_G = (pi alpha Z1 Z2)^2 * 2 mu c^2, written directly from the
    barrier-penetration integral as E_G = 2 mu c^2 (pi Z1 Z2 e^2 / (4 pi eps0 hbar c))^2.
    Returns joules. sqrt(E_G) is the constant b in the exp(-b/sqrt(E)) Gamow factor."""
    c = 2.99792458e8
    alpha_zz = Z1 * Z2 * E_CHARGE ** 2 / (4.0 * math.pi * EPS0 * HBAR * c)
    return 2.0 * mu * c ** 2 * (math.pi * alpha_zz) ** 2


def gamow_peak_energy(Z1: float, Z2: float, mu: float, T: float) -> float:
    """Gamow peak energy E0 = (E_G (k_B T)^2 / 4)^(1/3) (J): where the product of the
    Boltzmann tail and the tunnelling probability is maximal."""
    E_G = gamow_energy(Z1, Z2, mu)
    return (E_G * (K_B * T) ** 2 / 4.0) ** (1.0 / 3.0)


def gamow_peak_width(Z1: float, Z2: float, mu: float, T: float) -> float:
    """Full width (1/e) of the Gamow peak, delta = 4 sqrt(E0 k_B T / 3) (J)."""
    E0 = gamow_peak_energy(Z1, Z2, mu, T)
    return 4.0 * math.sqrt(E0 * K_B * T / 3.0)


def integrand(E: float, Z1: float, Z2: float, mu: float, T: float) -> float:
    """Relative reaction-rate integrand exp(-E/kT - sqrt(E_G/E)), the product of the
    Boltzmann factor and the Gamow tunnelling factor (unnormalized)."""
    E_G = gamow_energy(Z1, Z2, mu)
    return math.exp(-E / (K_B * T) - math.sqrt(E_G / E))


def peak_energy_kev(Z1: float, Z2: float, mu: float, T: float) -> float:
    """Gamow peak energy expressed in keV."""
    return gamow_peak_energy(Z1, Z2, mu, T) / KEV
