"""The Zeeman effect: splitting spectral lines with a magnetic field.

An atom's electron has a magnetic moment, so its energy levels shift in an applied magnetic
field -- and a single spectral line splits into several. Pieter Zeeman saw it in 1896, and it
became the tool for measuring magnetic fields on the Sun and in stars, and a first probe of
electron spin.

The energy shift of a sublevel is

    delta_E = g_J m_J mu_B B,

where mu_B = e hbar / (2 m_e) is the Bohr magneton, m_J the magnetic quantum number, and g_J
the Lande g-factor built from the orbital and spin angular momenta. In the *normal* Zeeman
effect (spin cancels, g = 1) the line splits into a clean triplet whose components are shifted
by

    delta_nu = mu_B B / h        ( = 14.0 GHz per tesla ),

with the middle line unshifted and the two others +/- delta_nu -- the Lorentz-triplet
classical physics predicted. The *anomalous* Zeeman effect (nonzero spin, g != 1) splits into
more, unevenly spaced lines, and its very existence was one of the puzzles that forced the
discovery of electron spin.

The Lande g-factor for a level with quantum numbers (J, L, S) is

    g_J = 1 + [J(J+1) + S(S+1) - L(L+1)] / [2 J(J+1)],

running from 1 (pure orbital) to 2 (pure spin). This module gives the Bohr magneton, the
normal-Zeeman frequency and wavelength splitting, the Lande g-factor, the anomalous sublevel
shift, and the field inferred from a measured splitting, and reproduces the 14 GHz/T normal
shift and the sodium D-line g-factors. SI units. Pure stdlib; the atomic-spectroscopy
companion to the Bohr and blackbody notes.
"""

from __future__ import annotations

E_CHARGE = 1.602176634e-19    # C
HBAR = 1.054571817e-34       # J s
H = 6.62607015e-34           # J s
M_E = 9.1093837015e-31       # kg
C = 299792458.0              # m/s

BOHR_MAGNETON = E_CHARGE * HBAR / (2.0 * M_E)     # J/T, ~9.274e-24


def bohr_magneton() -> float:
    """The Bohr magneton mu_B = e hbar / (2 m_e) ~ 9.274e-24 J/T, the natural unit of atomic
    magnetic moment."""
    return BOHR_MAGNETON


def normal_zeeman_shift_hz(field: float) -> float:
    """Frequency shift of a normal-Zeeman component, delta_nu = mu_B B / h (Hz).
    ~14.0 GHz per tesla -- the classical Lorentz-triplet spacing."""
    return BOHR_MAGNETON * field / H


def normal_zeeman_shift_wavelength(field: float, wavelength: float) -> float:
    """Wavelength shift |delta_lambda| = lambda^2 delta_nu / c (m) of a normal-Zeeman
    component at rest wavelength lambda."""
    return wavelength * wavelength * normal_zeeman_shift_hz(field) / C


def lande_g(j: float, l: float, s: float) -> float:
    """Lande g-factor g_J = 1 + [J(J+1)+S(S+1)-L(L+1)] / [2 J(J+1)], the coupling of orbital
    (L) and spin (S) moments. 1 for pure orbital, 2 for pure spin."""
    if j == 0:
        return 0.0
    return 1.0 + (j * (j + 1) + s * (s + 1) - l * (l + 1)) / (2.0 * j * (j + 1))


def energy_shift(g_j: float, m_j: float, field: float) -> float:
    """Energy shift delta_E = g_J m_J mu_B B (J) of a sublevel m_J in field B."""
    return g_j * m_j * BOHR_MAGNETON * field


def anomalous_shift_hz(g_j: float, m_j: float, field: float) -> float:
    """Frequency shift g_J m_J mu_B B / h (Hz) of an anomalous-Zeeman sublevel."""
    return energy_shift(g_j, m_j, field) / H


def field_from_splitting(delta_nu: float) -> float:
    """Magnetic field B = h delta_nu / mu_B (T) inferred from a measured normal-Zeeman
    frequency splitting -- how solar magnetograms read the Sun's field. Inverts the shift."""
    return H * delta_nu / BOHR_MAGNETON
