"""The Tully-Fisher relation: a spiral galaxy's brightness from how fast it spins.

Spiral galaxies obey a remarkably tight scaling between their luminosity and their
flat-rotation speed,

    L ~ v_flat^4,      or in magnitudes   M = a (log10 v_flat - 2.6) + b,

with a slope near -10 in the near-infrared. The tightness follows from combining the
circular-orbit condition v^2 = G M / R with a roughly constant surface brightness and
mass-to-light ratio: more massive spirals both spin faster and shine brighter, in
lockstep. The baryonic version (using total gas + stellar mass instead of light) is
even tighter, M_baryon ~ v^4, and is a key test of dark-matter and modified-gravity
models.

Because L depends only on the easily-measured rotation width (from the 21-cm line or
optical spectra), Tully-Fisher is a powerful redshift-independent distance indicator:
measure v_flat to get the absolute magnitude, compare with the apparent magnitude, and
read off the distance modulus. It extends the cosmic distance ladder well beyond where
individual Cepheids can be resolved.

This module gives the luminosity from rotation speed, the absolute magnitude via the
calibrated relation, the implied distance from an apparent magnitude, and the inverse
(rotation speed from luminosity), and reproduces the L ~ v^4 slope and a Milky-Way-like
spiral. Solar / astronomical units. Pure stdlib; the spiral-galaxy companion to the
Faber-Jackson and standard-candle modules.
"""

from __future__ import annotations

import math

# near-infrared Tully-Fisher: M = TF_SLOPE (log10 v - 2.6) + TF_ZP  (v in km/s)
TF_SLOPE = -9.5
TF_ZP = -21.0                  # absolute magnitude of a v=400 km/s ... anchored at log v=2.6
L_SUN = 3.828e26


def luminosity_from_vflat(v_flat_kms: float, v_ref: float = 200.0,
                          L_ref: float = 2e10, power: float = 4.0) -> float:
    """Luminosity (in solar luminosities) from the flat rotation speed via L ~ v^4,
    normalized so a v_ref=200 km/s spiral has L_ref=2e10 L_sun."""
    return L_ref * (v_flat_kms / v_ref) ** power


def absolute_magnitude(v_flat_kms: float, slope: float = TF_SLOPE,
                       zero_point: float = TF_ZP) -> float:
    """Absolute magnitude from the calibrated Tully-Fisher relation:
    M = slope (log10 v - 2.6) + zero_point. Faster rotators are brighter."""
    return slope * (math.log10(v_flat_kms) - 2.6) + zero_point


def distance_from_apparent(v_flat_kms: float, m_app: float,
                           slope: float = TF_SLOPE,
                           zero_point: float = TF_ZP) -> float:
    """Distance (pc) to a spiral from its rotation speed and apparent magnitude, via
    Tully-Fisher M and the distance modulus d = 10^((m - M + 5)/5)."""
    M = absolute_magnitude(v_flat_kms, slope, zero_point)
    return 10.0 ** ((m_app - M + 5.0) / 5.0)


def vflat_from_luminosity(L_lsun: float, v_ref: float = 200.0,
                          L_ref: float = 2e10, power: float = 4.0) -> float:
    """Invert L ~ v^4 to recover the flat rotation speed (km/s) from luminosity."""
    return v_ref * (L_lsun / L_ref) ** (1.0 / power)


def baryonic_mass_from_vflat(v_flat_kms: float, A: float = 50.0) -> float:
    """Baryonic Tully-Fisher: M_baryon (solar masses) = A v_flat^4, with v in km/s.
    A ~ 50 reproduces the observed relation (M ~ 1e11 at v ~ 220 km/s)."""
    return A * v_flat_kms ** 4
