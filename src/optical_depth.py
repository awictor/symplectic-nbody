"""Optical depth and radiative transfer: where a star's surface is.

As light travels through matter it is absorbed and scattered. The OPTICAL DEPTH

    tau = integral kappa rho dl = n sigma L

(kappa the opacity, rho density, or n number density and sigma cross-section)
counts how many mean free paths the light crosses. The transmitted fraction of a
background beam is

    I / I_0 = exp(-tau),

so tau << 1 is "optically thin" (you see through it) and tau >> 1 is "optically
thick" (you see only the surface). A star has no solid surface; its PHOTOSPHERE
-- the layer we see and assign the effective temperature -- is where the optical
depth measured inward reaches tau ~ 2/3 (the Eddington-Barbier result).

This module gives the optical depth, the transmitted fraction, the photon mean
free path, and a thin/thick verdict, and reproduces those regimes. SI units.
Pure stdlib.
"""

from __future__ import annotations

import math

PHOTOSPHERE_TAU = 2.0 / 3.0


def optical_depth(n: float, sigma: float, length: float) -> float:
    """Optical depth tau = n sigma L for number density n, cross-section sigma,
    path length L."""
    return n * sigma * length


def optical_depth_kappa(kappa: float, rho: float, length: float) -> float:
    """Optical depth from opacity: tau = kappa rho L."""
    return kappa * rho * length


def transmitted_fraction(tau: float) -> float:
    """Fraction of a background beam that survives: I/I0 = exp(-tau)."""
    return math.exp(-tau)


def mean_free_path(n: float, sigma: float) -> float:
    """Photon mean free path l = 1 / (n sigma)."""
    return 1.0 / (n * sigma)


def is_optically_thick(tau: float) -> bool:
    """Optically thick (see only the surface) when tau > 1."""
    return tau > 1.0


def photosphere_depth(n: float, sigma: float) -> float:
    """Geometric depth (from the outside) at which the inward optical depth
    reaches the photospheric value tau = 2/3: L = (2/3) / (n sigma)."""
    return PHOTOSPHERE_TAU / (n * sigma)
