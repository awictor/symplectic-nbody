"""Ram-pressure stripping: how galaxy clusters strip spirals of their gas.

A galaxy plunging through the hot, tenuous gas that fills a cluster (the
intracluster medium, ICM) feels a wind. That wind exerts a ram pressure

    P_ram = rho_icm v^2,

pushing on the galaxy's interstellar gas. The disk's own gravity holds that gas down
with a restoring force per unit area set by the stellar and gas surface densities.
The Gunn & Gott (1972) criterion says gas at galactocentric radius R is stripped when
the ram pressure exceeds the gravitational restoring force per area:

    rho_icm v^2 > 2 pi G Sigma_star Sigma_gas,

so a galaxy keeps only the gas inside the "stripping radius" where its self-gravity
still wins. Because P_ram ~ rho v^2 and clusters are both dense and dynamically hot
(v ~ 1000-2000 km/s), infalling spirals can be stripped of much of their gas on a
single pass -- quenching star formation and helping turn field spirals into the
gas-poor S0 and elliptical galaxies that dominate cluster cores.

This module gives the ram pressure, the gravitational restoring pressure, the
stripping verdict, and the surviving gas radius for an exponential disk, and
reproduces the strong stripping of a Milky-Way-like galaxy falling into a rich
cluster. SI units. Pure stdlib; the cluster-environment companion to the galaxy and
cluster modules.
"""

from __future__ import annotations

import math

G = 6.67430e-11
KM = 1e3
KPC = 3.0856775814913673e19    # kiloparsec (m)
M_SUN = 1.989e30
M_P = 1.6726219e-27
MYR = 3.15576e13


def ram_pressure(rho_icm: float, v: float) -> float:
    """Ram pressure P = rho_icm v^2 (Pa) on a galaxy moving at speed v through ICM
    of mass density rho_icm."""
    return rho_icm * v * v


def icm_density(n_per_cc: float, mu: float = 0.6) -> float:
    """ICM mass density (kg/m^3) from electron/particle number density n (per cc):
    rho = mu m_p n. Typical cluster core n ~ 1e-3 /cc."""
    return mu * M_P * n_per_cc * 1e6


def restoring_pressure(sigma_star: float, sigma_gas: float) -> float:
    """Maximum gravitational restoring pressure per area holding the disk gas down:
    P_grav = 2 pi G Sigma_star Sigma_gas (Pa), with surface densities in kg/m^2."""
    return 2.0 * math.pi * G * sigma_star * sigma_gas


def is_stripped(rho_icm: float, v: float, sigma_star: float,
                sigma_gas: float) -> bool:
    """Gunn-Gott: True if ram pressure overcomes the disk's gravitational hold,
    rho_icm v^2 > 2 pi G Sigma_star Sigma_gas."""
    return ram_pressure(rho_icm, v) > restoring_pressure(sigma_star, sigma_gas)


def stripping_radius(rho_icm: float, v: float, sigma0_star: float,
                     sigma0_gas: float, scale_length: float) -> float:
    """Radius (m) outside which gas is stripped, for exponential disks
    Sigma(R) = Sigma0 exp(-R/h). Gas survives where 2 pi G Sigma_star Sigma_gas >
    rho v^2; with both disks sharing scale length h the product ~ exp(-2R/h), so

        R_strip = (h/2) ln( 2 pi G Sigma0_star Sigma0_gas / (rho v^2) ).

    Returns 0 if even the centre is stripped, inf if nothing is."""
    P_ram = ram_pressure(rho_icm, v)
    P0 = restoring_pressure(sigma0_star, sigma0_gas)
    if P_ram >= P0:
        return 0.0
    return 0.5 * scale_length * math.log(P0 / P_ram)
