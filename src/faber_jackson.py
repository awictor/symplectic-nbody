"""The Faber-Jackson relation: elliptical galaxies' brightness from their motion.

Elliptical galaxies obey a tight scaling between luminosity and the random
stellar velocity dispersion sigma:

    L ~ sigma^4     (Faber & Jackson 1976),

the elliptical-galaxy analog of the Tully-Fisher relation for spirals. It follows
from the virial theorem plus a roughly constant mass-to-light ratio and surface
brightness: virial mass M ~ sigma^2 R / G, and with M/L and surface brightness
fixed, L ~ sigma^4.

Because L depends steeply on sigma, a measured velocity dispersion (from spectral
line widths) gives the galaxy's luminosity -- hence, comparing to the apparent
brightness, its distance. This module gives the Faber-Jackson luminosity, the
virial mass, and the mass-to-light ratio, and reproduces the sigma^4 slope.
Solar units. Pure stdlib; complements virial and cluster.
"""

from __future__ import annotations

import math

G = 6.67430e-11
M_SUN = 1.98892e30
L_SUN = 3.828e26
KM = 1000.0
KPC = 3.0857e19

# Faber-Jackson normalization: L ~ 2e10 L_sun at sigma = 200 km/s (typical L*).
SIGMA_STAR = 200e3            # m/s
L_STAR = 2e10                 # L_sun


def faber_jackson_luminosity(sigma: float) -> float:
    """Elliptical-galaxy luminosity (L_sun) from velocity dispersion sigma (m/s):
    L = L_star (sigma / sigma_star)^4."""
    return L_STAR * (sigma / SIGMA_STAR) ** 4


def virial_mass(sigma: float, R: float) -> float:
    """Virial mass estimate M ~ sigma^2 R / G (kg) for dispersion sigma and
    effective radius R."""
    return sigma * sigma * R / G


def mass_to_light(sigma: float, R: float) -> float:
    """Mass-to-light ratio M/L in solar units (M_sun / L_sun)."""
    M = virial_mass(sigma, R) / M_SUN
    L = faber_jackson_luminosity(sigma)
    return M / L


def dispersion_from_luminosity(L_lsun: float) -> float:
    """Invert Faber-Jackson: sigma = sigma_star (L / L_star)^{1/4} (m/s)."""
    return SIGMA_STAR * (L_lsun / L_STAR) ** 0.25
