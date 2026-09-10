"""Weighing a galaxy cluster from how fast its members move.

You cannot put a cluster of galaxies on a scale, but the virial theorem lets its own motion
weigh it. For a bound, relaxed self-gravitating system 2T + U = 0, and writing the kinetic
energy from the velocity dispersion and the potential from the size gives the virial mass
estimator

    M = alpha sigma^2 R / G,

where sigma is the velocity dispersion, R a characteristic radius, and alpha an order-unity
structure factor (~5 for the 3-D dispersion and a Plummer-like profile). In practice you only
measure the *line-of-sight* dispersion sigma_los from Doppler shifts, and for an isotropic
system sigma^2 = 3 sigma_los^2, so

    M ~ 5 (3 sigma_los^2) R / G.

This is exactly the calculation Zwicky did on the Coma cluster in 1933: its galaxies moved so
fast (sigma_los ~ 1000 km/s across R ~ 1-2 Mpc) that the mass needed to bind them dwarfed the
mass of the visible stars by a factor of ~100. Dividing the dynamical mass by the luminosity
gives a mass-to-light ratio far above the ~few of stellar populations -- the first evidence
for dark matter.

This module gives the virial mass from the 3-D or line-of-sight dispersion, the escape
velocity, the crossing time, the mass-to-light ratio, and the dark-matter fraction implied by
comparing the dynamical mass to the stellar mass, and reproduces Coma's ~10^15 solar-mass
dynamical mass and its large M/L. SI units internally; helpers accept km/s and Mpc. Pure
stdlib; the extragalactic-dynamics companion to the virial and Jeans notes.
"""

from __future__ import annotations

import math

G = 6.674e-11                  # gravitational constant (SI)
M_SUN = 1.989e30              # kg
MPC = 3.0857e22               # metres per megaparsec
KM = 1000.0                   # metres per km
L_SUN = 3.828e26             # watts (solar luminosity, for M/L bookkeeping)


def virial_mass(sigma_3d: float, radius: float, alpha: float = 5.0) -> float:
    """Virial mass M = alpha sigma^2 R / G (kg) from the 3-D velocity dispersion sigma (m/s)
    and characteristic radius R (m). alpha ~ 5 for a typical profile."""
    return alpha * sigma_3d * sigma_3d * radius / G


def virial_mass_los(sigma_los_kms: float, radius_mpc: float, alpha: float = 5.0) -> float:
    """Virial mass (kg) from the observed line-of-sight dispersion (km/s) and radius (Mpc),
    assuming isotropy so sigma^2 = 3 sigma_los^2. The practical estimator."""
    sigma_3d = math.sqrt(3.0) * sigma_los_kms * KM
    return virial_mass(sigma_3d, radius_mpc * MPC, alpha)


def escape_velocity(mass: float, radius: float) -> float:
    """Escape velocity sqrt(2 G M / R) (m/s) at the edge of a mass M within radius R."""
    return math.sqrt(2.0 * G * mass / radius)


def crossing_time(radius: float, sigma_3d: float) -> float:
    """Crossing time R / sigma (s): how long a galaxy takes to traverse the cluster. Much
    shorter than the age of the universe means the system has had time to relax."""
    return radius / sigma_3d


def mass_to_light(mass: float, luminosity_lsun: float) -> float:
    """Mass-to-light ratio in solar units (M/M_sun per L/L_sun): (M/M_sun) / (L/L_sun).
    A few for stars; hundreds for a cluster, the signature of dark matter."""
    return (mass / M_SUN) / luminosity_lsun


def dark_matter_fraction(dynamical_mass: float, luminous_mass: float) -> float:
    """Fraction of the total (dynamical) mass that is NOT luminous:
    1 - M_luminous / M_dynamical. ~0.9+ for clusters."""
    return 1.0 - luminous_mass / dynamical_mass
