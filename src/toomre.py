"""The Toomre Q criterion: when a rotating disk fragments into clumps and arms.

A thin, differentially rotating disk of gas or stars is caught between three
influences. Self-gravity tries to collapse an overdense patch; random motions
(pressure, or velocity dispersion) puff it back apart on small scales; and the
disk's rotation (through the Coriolis force, measured by the epicyclic frequency
kappa) resists collapse on large scales. Toomre (1964) showed the balance is set by
one dimensionless number,

    Q = c_s kappa / (pi G Sigma)      (gas disk),
    Q = sigma_R kappa / (3.36 G Sigma)  (stellar disk),

where c_s (or sigma_R) is the sound speed / radial velocity dispersion, Sigma the
surface density, and kappa the epicyclic frequency. When Q > 1 the disk is stable:
rotation and pressure between them defeat gravity at every wavelength. When Q < 1 a
band of intermediate wavelengths goes unstable and the disk fragments into rings,
clumps and spiral arms.

The most unstable wavelength is the Toomre wavelength lambda_T = 4 pi^2 G Sigma /
kappa^2 (equivalently 2 pi^2 G Sigma / kappa^2 for the critical case), which sets the
characteristic size of the structures that grow. The solar neighbourhood sits at
Q ~ 1.5-2 -- marginally stable, which is exactly why the Milky Way has spiral arms
but has not collapsed wholesale into giant clumps.

This module gives the gas and stellar Q, the stability verdict, and the Toomre
wavelength, and reproduces the marginally-stable solar neighbourhood. SI units. Pure
stdlib; the rotating-disk sibling of the Jeans-collapse module.
"""

from __future__ import annotations

import math

G = 6.67430e-11
PC = 3.0856775814913673e16      # parsec (m)
M_SUN = 1.989e30
KM = 1e3
STELLAR_CONST = 3.36            # coefficient in the stellar-disk Toomre Q


def toomre_q_gas(c_s: float, kappa: float, sigma: float) -> float:
    """Gas-disk Toomre parameter Q = c_s kappa / (pi G Sigma). Q>1 stable, Q<1 not."""
    return c_s * kappa / (math.pi * G * sigma)


def toomre_q_stars(sigma_R: float, kappa: float, sigma: float) -> float:
    """Stellar-disk Toomre parameter Q = sigma_R kappa / (3.36 G Sigma)."""
    return sigma_R * kappa / (STELLAR_CONST * G * sigma)


def is_stable(Q: float) -> bool:
    """True if the disk is Toomre-stable (Q >= 1): rotation + pressure beat gravity."""
    return Q >= 1.0


def toomre_wavelength(sigma: float, kappa: float) -> float:
    """Most-unstable Toomre wavelength lambda_T = 4 pi^2 G Sigma / kappa^2 (m).
    Sets the characteristic size of the clumps/arms that grow when Q < 1."""
    return 4.0 * math.pi ** 2 * G * sigma / (kappa ** 2)


def critical_dispersion_gas(kappa: float, sigma: float) -> float:
    """Sound speed at which a gas disk is marginally stable (Q = 1):
    c_s = pi G Sigma / kappa. Below this the disk fragments."""
    return math.pi * G * sigma / kappa


def epicyclic_frequency_flat(v_circ: float, R: float) -> float:
    """Epicyclic frequency kappa for a flat rotation curve (v_circ constant):
    kappa = sqrt(2) v_circ / R (since kappa = sqrt(2) Omega for a flat curve)."""
    return math.sqrt(2.0) * v_circ / R
