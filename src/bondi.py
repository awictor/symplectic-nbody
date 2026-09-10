"""Bondi accretion: how a compact object swallows the gas around it.

A gravitating body at rest in a gas cloud pulls material in spherically. Gas
within the BONDI RADIUS, where the escape speed exceeds the sound speed,

    r_B = G M / c_s^2,

falls in, giving the steady Bondi (1952) accretion rate

    Mdot = 4 pi lambda (G M)^2 rho_inf / c_s^3,

with lambda ~ 0.25 for a gamma = 5/3 gas. The rate scales as M^2 (bigger holes
eat faster, so accretion runs away), linearly with the ambient density, and as
c_s^{-3} (cold gas is much easier to accrete). Compared against the Eddington
rate it tells you whether an object grows freely or is radiation-limited.

This module gives the Bondi radius, the accretion rate, and the accretion
luminosity, and reproduces the expected scalings. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.98892e30
YEAR = 3.15576e7
LAMBDA_53 = 0.25               # accretion eigenvalue for gamma = 5/3


def bondi_radius(M: float, cs: float) -> float:
    """Bondi radius r_B = G M / c_s^2 (m)."""
    return G * M / (cs * cs)


def bondi_rate(M: float, rho_inf: float, cs: float,
               lam: float = LAMBDA_53) -> float:
    """Steady Bondi accretion rate Mdot = 4 pi lambda (G M)^2 rho / c_s^3 (kg/s)."""
    return 4.0 * math.pi * lam * (G * M) ** 2 * rho_inf / cs ** 3


def accretion_luminosity(Mdot: float, efficiency: float = 0.1) -> float:
    """Accretion luminosity L = eta Mdot c^2 (W)."""
    return efficiency * Mdot * C * C


def sound_speed(T: float, mu: float = 1.0, gamma: float = 5.0 / 3.0) -> float:
    """Adiabatic sound speed c_s = sqrt(gamma k T / (mu m_p)) (m/s)."""
    K_B = 1.380649e-23
    M_P = 1.6726219e-27
    return math.sqrt(gamma * K_B * T / (mu * M_P))
