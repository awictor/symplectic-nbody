"""Moseley's law: putting the periodic table in order by X-ray colour.

Fire fast electrons at an element and it fluoresces characteristic X-rays: an electron knocked
out of the innermost (K) shell is replaced by one falling from above, emitting a photon whose
energy is a fingerprint of the element. In 1913 Henry Moseley measured these lines across the
elements and found a stunningly simple pattern -- the square root of the frequency rises
*linearly* with atomic number:

    sqrt(f) = a (Z - sigma),

with sigma a small screening constant (~1 for the K-alpha line, because the one remaining K
electron screens the nucleus). Equivalently the K-alpha photon energy is a Rydberg-like

    E_Kalpha = 13.6 eV * (3/4) * (Z - 1)^2,

from the n=2 -> n=1 transition seen by a nearly-unscreened nuclear charge (Z-1). Moseley's law
did what atomic weight could not: it ordered the elements by nuclear charge Z, exposed gaps
where undiscovered elements (technetium, promethium) had to sit, and proved Z -- not weight --
is the true atomic serial number. It is still how an electron microprobe or XRF gun identifies
which elements a sample contains.

This module gives the K-alpha energy, wavelength and frequency for an element, the atomic
number inferred from a measured line (elemental identification), and the general screened
transition energy, and reproduces copper's 8.05 keV K-alpha and the linear sqrt(f)-vs-Z law.
SI units with eV/keV helpers. Pure stdlib; the atomic-spectroscopy companion to the Bohr and
Franck-Hertz notes.
"""

from __future__ import annotations

import math

RYDBERG_EV = 13.605693        # Rydberg energy (eV)
E_CHARGE = 1.602176634e-19
H = 6.62607015e-34
C = 299792458.0


def k_alpha_energy_ev(z: int) -> float:
    """K-alpha photon energy (eV) = 13.6 (3/4) (Z-1)^2 for the n=2->1 transition with the
    nuclear charge screened to (Z-1). ~8050 eV for copper (Z=29)."""
    return RYDBERG_EV * 0.75 * (z - 1) ** 2


def k_alpha_frequency(z: int) -> float:
    """K-alpha frequency (Hz) = E / h."""
    return k_alpha_energy_ev(z) * E_CHARGE / H


def k_alpha_wavelength(z: int) -> float:
    """K-alpha wavelength (m) = h c / E. ~0.154 nm for copper (the standard XRD source)."""
    return H * C / (k_alpha_energy_ev(z) * E_CHARGE)


def sqrt_frequency(z: int) -> float:
    """sqrt of the K-alpha frequency (Hz^1/2): Moseley's law says this is linear in Z."""
    return math.sqrt(k_alpha_frequency(z))


def atomic_number_from_energy(energy_ev: float) -> float:
    """Atomic number Z inferred from a measured K-alpha energy: Z = 1 + sqrt(E / (13.6*3/4)).
    How XRF/microprobe identifies an element. Real-valued; round to the nearest integer."""
    return 1.0 + math.sqrt(energy_ev / (RYDBERG_EV * 0.75))


def transition_energy_ev(z: int, n_lower: int, n_upper: int, screening: float = 1.0) -> float:
    """Screened hydrogenic transition energy (eV) = 13.6 (Z - sigma)^2 (1/n_low^2 - 1/n_up^2).
    K-alpha is n_lower=1, n_upper=2, screening ~1; sets other X-ray lines (K-beta, L series)."""
    return RYDBERG_EV * (z - screening) ** 2 * (1.0 / n_lower ** 2 - 1.0 / n_upper ** 2)
