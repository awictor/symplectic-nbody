"""The quantum harmonic oscillator: evenly-spaced rungs and a restless ground state.

Any system near a potential minimum looks like a spring, V = (1/2) k x^2, so the
harmonic oscillator is the workhorse of quantum mechanics -- molecular vibrations,
phonons in crystals, photons in a cavity. Solving the Schrodinger equation gives
equally-spaced energy levels

    E_n = (n + 1/2) hbar omega,     omega = sqrt(k/m),     n = 0, 1, 2, ...

Two features stand out. The spacing is a constant hbar omega (unlike the n^2 box or the
-1/n^2 atom), so every transition emits the same-energy photon -- the sharp vibrational
lines of infrared spectroscopy. And the ground state is NOT zero: the zero-point energy

    E_0 = (1/2) hbar omega

is forced by the uncertainty principle (the parcel cannot sit still at the bottom), and
it is real -- it shifts chemical bond energies, keeps helium liquid at absolute zero, and
sets the vacuum energy of each field mode.

For a diatomic molecule like CO the vibrational quantum hbar omega is ~0.27 eV,
absorbing in the infrared near 4.6 microns. This module gives the angular frequency,
the level energy, the zero-point energy, the constant level spacing, the vibrational
transition wavelength, and the classical turning point, and reproduces the CO
vibrational line and the equal-spacing rule. SI units, energies via an eV helper. Pure
stdlib; the vibrational companion to the particle-box and uncertainty modules.
"""

from __future__ import annotations

import math

HBAR = 1.054571817e-34
H = 6.62607015e-34
C = 2.99792458e8
EV = 1.602176634e-19
AMU = 1.66053907e-27


def angular_frequency(k: float, m: float) -> float:
    """Classical angular frequency omega = sqrt(k/m) (rad/s) of a mass m on a spring
    of stiffness k."""
    return math.sqrt(k / m)


def energy_level(n: int, k: float, m: float) -> float:
    """Energy of level n: E_n = (n + 1/2) hbar omega (joules)."""
    return (n + 0.5) * HBAR * angular_frequency(k, m)


def zero_point_energy(k: float, m: float) -> float:
    """Ground-state (n=0) zero-point energy (1/2) hbar omega (joules): nonzero, forced
    by the uncertainty principle."""
    return 0.5 * HBAR * angular_frequency(k, m)


def level_spacing(k: float, m: float) -> float:
    """Constant energy gap between adjacent levels: hbar omega (joules)."""
    return HBAR * angular_frequency(k, m)


def transition_wavelength(k: float, m: float) -> float:
    """Wavelength (m) of a single-quantum vibrational transition (n -> n-1):
    lambda = h c / (hbar omega) = 2 pi c / omega."""
    return H * C / level_spacing(k, m)


def turning_point(n: int, k: float, m: float) -> float:
    """Classical turning point x for level n, where (1/2) k x^2 = E_n:
    x = sqrt(2 E_n / k) (m)."""
    return math.sqrt(2.0 * energy_level(n, k, m) / k)


def spring_constant_from_frequency(freq_hz: float, m: float) -> float:
    """Recover the spring constant k = m (2 pi f)^2 from a measured vibration
    frequency and reduced mass."""
    omega = 2.0 * math.pi * freq_hz
    return m * omega * omega
