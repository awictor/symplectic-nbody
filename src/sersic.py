"""The Sersic profile: the shape of a galaxy's light.

Photograph a galaxy and its surface brightness falls off from the centre in a remarkably
regular way. Sersic (1963) captured it with a one-parameter family,

    I(R) = I_e exp{ -b_n [ (R/R_e)^(1/n) - 1 ] },

where R_e is the *effective radius* enclosing half the total light, I_e the brightness there,
and n the Sersic index that sets how centrally concentrated the profile is. The constant b_n
is fixed by the half-light definition and is very well approximated by

    b_n = 2n - 1/3 + 0.009876/n.

Two values of n are famous. n = 1 is the exponential disk of a spiral galaxy, I ~ exp(-R/h)
with scale length h = R_e/1.678. n = 4 is the de Vaucouleurs law, I ~ exp(-R^(1/4)), the
steep, extended profile of a giant elliptical or a classical bulge -- a bright core and
enormous faint wings. Larger n means a more concentrated core and a more extended halo of
light at once.

Integrating the profile gives the total luminosity in closed form via the gamma function,

    L_tot = I_e R_e^2 2 pi n e^{b_n} Gamma(2n) / b_n^{2n},

so measuring I_e, R_e and n weighs a galaxy's stars. This module gives the surface brightness
at any radius, b_n, the total luminosity, the exponential scale length, and the fraction of
light enclosed within a radius, and reproduces the half-light property (half the luminosity
inside R_e) and the n=1 exponential limit. Magnitudes-per-arcsec^2 helpers included. Pure
stdlib (gamma/gammainc by series); the galaxy-structure companion to the cluster-mass and
Tully-Fisher notes.
"""

from __future__ import annotations

import math


def b_n(n: float) -> float:
    """Sersic b_n, the constant tying R_e to the half-light radius:
    b_n ~ 2n - 1/3 + 0.009876/n (Ciotti & Bertin). Accurate for n >~ 0.5."""
    return 2.0 * n - 1.0 / 3.0 + 0.009876 / n


def surface_brightness(radius: float, i_e: float, r_e: float, n: float) -> float:
    """Sersic surface brightness I(R) = I_e exp{-b_n[(R/R_e)^(1/n) - 1]} (same units as I_e).
    Equals I_e at R = R_e by construction."""
    bn = b_n(n)
    return i_e * math.exp(-bn * ((radius / r_e) ** (1.0 / n) - 1.0))


def total_luminosity(i_e: float, r_e: float, n: float) -> float:
    """Total luminosity from integrating the profile over the plane:
    L = I_e R_e^2 2 pi n e^{b_n} Gamma(2n) / b_n^{2n}."""
    bn = b_n(n)
    return i_e * r_e * r_e * 2.0 * math.pi * n * math.exp(bn) * math.gamma(2.0 * n) / bn ** (2.0 * n)


def exponential_scale_length(r_e: float) -> float:
    """For an n=1 exponential disk, the scale length h = R_e / b_1 = R_e / 1.678 (m or same
    units as R_e), so I ~ exp(-R/h)."""
    return r_e / b_n(1.0)


def _gammainc_lower_regularized(s: float, x: float) -> float:
    """Regularized lower incomplete gamma P(s, x) = gamma(s, x)/Gamma(s), by series (good for
    x up to ~s+40). Used for the enclosed-light fraction."""
    if x <= 0.0:
        return 0.0
    # series: x^s e^-x sum_{k=0}^inf x^k / (s (s+1) ... (s+k))
    term = 1.0 / s
    total = term
    for k in range(1, 1000):
        term *= x / (s + k)
        total += term
        if term < 1e-14 * total:
            break
    return total * math.exp(s * math.log(x) - x - math.lgamma(s))


def enclosed_light_fraction(radius: float, r_e: float, n: float) -> float:
    """Fraction of the total luminosity within projected radius R:
    P(2n, b_n (R/R_e)^(1/n)). Equals 1/2 at R = R_e by definition of the effective radius."""
    bn = b_n(n)
    x = bn * (radius / r_e) ** (1.0 / n)
    return _gammainc_lower_regularized(2.0 * n, x)


def brightness_to_mag(i_ratio: float, mu_e: float) -> float:
    """Surface brightness in mag/arcsec^2 from an intensity ratio I/I_e and the effective
    surface brightness mu_e: mu = mu_e - 2.5 log10(I/I_e). Fainter = larger mu."""
    return mu_e - 2.5 * math.log10(i_ratio)
