"""The Shakura-Sunyaev accretion disk: why black holes glow in X-rays.

Gas cannot fall straight into a compact object -- it carries angular momentum, so it
settles into a disk and spirals inward only as viscosity ferries angular momentum
outward. Each ring, orbiting at the local Keplerian rate, rubs against its neighbours
and dissipates gravitational energy as heat, which the disk radiates as a blackbody.

Balancing the released gravitational power against blackbody emission gives the
classic disk temperature profile

    T(r) = T_* (r / R_in)^(-3/4)   far from the inner edge,

falling as r^(-3/4). The inner edge (the innermost stable circular orbit, ~3
Schwarzschild radii for a non-spinning black hole) is the hottest point, so a
stellar-mass black hole accreting near its limit peaks in soft X-rays while a
supermassive one peaks in the ultraviolet -- exactly the "big blue bump" seen in
quasars.

Two more numbers close the picture. The Eddington luminosity L_Edd = 4 pi G M m_p c
/ sigma_T caps the steady accretion rate (radiation pressure would otherwise blow the
inflow away), and the accretion efficiency eta ~ 0.057 for a Schwarzschild hole (up
to ~0.42 for a maximal Kerr hole) sets how much rest-mass energy L = eta Mdot c^2 the
disk converts to light -- far more than the ~0.007 of hydrogen fusion, which is why
accretion onto black holes powers the brightest steady sources in the universe.

This module gives the disk temperature profile, the characteristic and peak
temperatures, the Eddington luminosity, and the radiative efficiency, and reproduces
the X-ray-hot stellar-mass disk and UV-bright quasar. SI units. Pure stdlib; built on
the blackbody, Eddington, and Schwarzschild pieces.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
SIGMA = 5.670374419e-8         # Stefan-Boltzmann
M_P = 1.6726219e-27
SIGMA_T = 6.6524587e-29        # Thomson cross section
M_SUN = 1.989e30

ETA_SCHWARZSCHILD = 0.0572     # radiative efficiency of a non-spinning black hole
ISCO_FACTOR = 3.0              # ISCO = 3 Schwarzschild radii (6 GM/c^2) for a=0


def schwarzschild_radius(M: float) -> float:
    """Schwarzschild radius r_s = 2 G M / c^2 (m)."""
    return 2.0 * G * M / (C * C)


def isco_radius(M: float) -> float:
    """Innermost stable circular orbit for a Schwarzschild hole: r_isco = 3 r_s
    = 6 G M / c^2 (m). The disk's hot inner edge."""
    return ISCO_FACTOR * schwarzschild_radius(M)


def disk_temperature(r: float, M: float, mdot: float, r_in: float = None) -> float:
    """Effective disk temperature (K) at radius r for accretion rate mdot (kg/s):
    from 3 G M mdot / (8 pi sigma r^3) with the (1 - sqrt(r_in/r)) inner boundary
    factor, so T ~ r^(-3/4) far out and T -> 0 at the inner edge."""
    if r_in is None:
        r_in = isco_radius(M)
    f = max(0.0, 1.0 - math.sqrt(r_in / r))
    flux = 3.0 * G * M * mdot / (8.0 * math.pi * SIGMA * r ** 3) * f
    return flux ** 0.25


def characteristic_temperature(M: float, mdot: float, r_in: float = None) -> float:
    """Characteristic temperature scale T_* = (3 G M mdot / 8 pi sigma r_in^3)^(1/4)
    (no boundary factor): the order of the peak disk temperature."""
    if r_in is None:
        r_in = isco_radius(M)
    return (3.0 * G * M * mdot / (8.0 * math.pi * SIGMA * r_in ** 3)) ** 0.25


def peak_temperature(M: float, mdot: float, r_in: float = None) -> float:
    """Actual maximum of T(r), which occurs at r = (49/36) r_in for the profile with
    the boundary factor: T_max ~ 0.488 T_*."""
    if r_in is None:
        r_in = isco_radius(M)
    r_peak = (49.0 / 36.0) * r_in
    return disk_temperature(r_peak, M, mdot, r_in)


def eddington_luminosity(M: float) -> float:
    """Eddington luminosity L_Edd = 4 pi G M m_p c / sigma_T (W)."""
    return 4.0 * math.pi * G * M * M_P * C / SIGMA_T


def radiative_efficiency(eta: float = ETA_SCHWARZSCHILD) -> float:
    """Accretion efficiency eta in L = eta Mdot c^2 (default Schwarzschild ~0.057)."""
    return eta


def luminosity(mdot: float, eta: float = ETA_SCHWARZSCHILD) -> float:
    """Accretion luminosity L = eta Mdot c^2 (W)."""
    return eta * mdot * C * C
