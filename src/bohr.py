"""The Bohr model: the hydrogen spectrum from a quantized orbit.

Bohr's 1913 leap was to postulate that an electron orbits the proton only in states of
quantized angular momentum, L = n hbar. Balancing the Coulomb attraction against the
centripetal requirement then fixes everything:

    radius        r_n = n^2 a_0,          a_0 = 4 pi eps0 hbar^2 / (m_e e^2) ~ 52.9 pm
    energy        E_n = -13.6 eV / n^2,
    photon        1/lambda = R_H (1/n1^2 - 1/n2^2),   R_H ~ 1.097e7 /m.

The ground state sits at -13.6 eV (the ionization energy), the Bohr radius a_0 sets the
atom's ~0.1 nm size, and transitions between levels emit the discrete spectral lines
that had baffled 19th-century physics: the Lyman series falls to n=1 (ultraviolet), the
Balmer series to n=2 (visible -- H-alpha at 656.3 nm, the red of nebulae), the Paschen
series to n=3 (infrared). Though superseded by full quantum mechanics, the Bohr model
gets the hydrogen energies and the Rydberg formula exactly right.

This module gives the level energy and radius, the transition wavelength via the Rydberg
formula, the ionization energy, and the orbital speed (whose ratio to c is the
fine-structure constant), and reproduces the 13.6 eV binding, the 52.9 pm Bohr radius,
and the 656.3 nm H-alpha line. SI units, energies via an eV helper. Pure stdlib; the
atomic-physics companion to the Saha and de-Broglie modules.
"""

from __future__ import annotations

import math

M_E = 9.1093837015e-31
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
HBAR = 1.054571817e-34
H = 6.62607015e-34
C = 2.99792458e8
EV = 1.602176634e-19

BOHR_RADIUS = 4.0 * math.pi * EPS0 * HBAR ** 2 / (M_E * E_CHARGE ** 2)  # ~5.29e-11 m
RYDBERG_ENERGY_EV = 13.605693  # ground-state binding energy magnitude
RYDBERG_CONSTANT = 1.0973731568e7  # R_H (m^-1)


def energy_level_ev(n: int) -> float:
    """Bohr energy of level n in eV: E_n = -13.6 / n^2 (bound, negative)."""
    return -RYDBERG_ENERGY_EV / (n * n)


def radius(n: int) -> float:
    """Orbital radius of level n (m): r_n = n^2 a_0."""
    return n * n * BOHR_RADIUS


def transition_wavelength(n1: int, n2: int) -> float:
    """Wavelength (m) of a transition between levels n1 < n2, via the Rydberg formula
    1/lambda = R_H (1/n1^2 - 1/n2^2). n1 is the lower level (final for emission)."""
    inv = RYDBERG_CONSTANT * (1.0 / (n1 * n1) - 1.0 / (n2 * n2))
    return 1.0 / inv


def transition_energy_ev(n1: int, n2: int) -> float:
    """Photon energy (eV) of the n2 -> n1 transition: E_n2 - E_n1 in magnitude."""
    return abs(energy_level_ev(n2) - energy_level_ev(n1))


def ionization_energy_ev(n: int = 1) -> float:
    """Energy (eV) to ionize from level n: |E_n| = 13.6 / n^2."""
    return abs(energy_level_ev(n))


def orbital_speed(n: int) -> float:
    """Electron orbital speed in level n (m/s): v_n = e^2 / (4 pi eps0 hbar n) = alpha c / n.
    For n=1, v/c = the fine-structure constant ~ 1/137."""
    return E_CHARGE ** 2 / (4.0 * math.pi * EPS0 * HBAR * n)
