"""The particle in a box: the simplest quantized system.

Trap a particle in an infinite square well of width L and its wavefunction must vanish
at the walls, so only standing waves with a whole number of half-wavelengths fit:
L = n lambda/2. Through the de Broglie relation p = h/lambda this quantizes the energy,

    E_n = n^2 h^2 / (8 m L^2),        n = 1, 2, 3, ...

The levels rise as n^2, the ground state (n=1) is nonzero (a zero-point energy forced by
confinement, exactly the uncertainty-principle scale), and squeezing the box (smaller L)
pushes every level up as 1/L^2. This 1/L^2 dependence is why quantum dots -- nanoscale
boxes for electrons -- glow at colours you can tune by size: a smaller dot has a wider
level gap and emits bluer light.

The wavefunctions psi_n(x) = sqrt(2/L) sin(n pi x/L) are orthonormal, with n-1 nodes.
Transitions between levels emit or absorb photons of energy E_n2 - E_n1, giving the
discrete absorption lines of a quantum well.

This module gives the energy levels, the level spacing, the transition wavelength, the
wavefunction, and the ground-state (zero-point) energy, and reproduces the eV-scale
gaps of nanometre electron boxes. SI units, energies via an eV helper. Pure stdlib; the
bound-state companion to the uncertainty and de-Broglie modules.
"""

from __future__ import annotations

import math

H = 6.62607015e-34
M_E = 9.1093837015e-31
EV = 1.602176634e-19
C = 2.99792458e8


def energy_level(n: int, L: float, m: float = M_E) -> float:
    """Energy of level n in an infinite square well of width L (joules):
    E_n = n^2 h^2 / (8 m L^2)."""
    return n * n * H * H / (8.0 * m * L * L)


def ground_state_energy(L: float, m: float = M_E) -> float:
    """Ground-state (n=1) zero-point energy (joules): the irreducible energy of
    confinement, h^2 / (8 m L^2)."""
    return energy_level(1, L, m)


def level_spacing(n: int, L: float, m: float = M_E) -> float:
    """Energy gap between levels n and n+1 (joules): E_{n+1} - E_n = (2n+1) E_1."""
    return energy_level(n + 1, L, m) - energy_level(n, L, m)


def transition_wavelength(n1: int, n2: int, L: float, m: float = M_E) -> float:
    """Photon wavelength (m) for the n2 -> n1 transition: lambda = h c / (E_n2 - E_n1)."""
    dE = abs(energy_level(n2, L, m) - energy_level(n1, L, m))
    return H * C / dE


def wavefunction(n: int, x: float, L: float) -> float:
    """Normalized eigenfunction psi_n(x) = sqrt(2/L) sin(n pi x / L) (m^-1/2),
    zero outside [0, L]."""
    if x < 0.0 or x > L:
        return 0.0
    return math.sqrt(2.0 / L) * math.sin(n * math.pi * x / L)


def box_width_for_gap(gap_ev: float, m: float = M_E) -> float:
    """Box width (m) whose n=1->2 gap equals a target energy: 3 E_1 = gap, so
    L = sqrt(3 h^2 / (8 m gap))."""
    gap = gap_ev * EV
    return math.sqrt(3.0 * H * H / (8.0 * m * gap))
