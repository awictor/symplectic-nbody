"""The Kerr black hole: rotation, frame-dragging, and the spin-dependent ISCO.

A rotating (Kerr) black hole of mass M and spin parameter a = J/M (0 <= a <= M,
in units G = c = 1) differs from Schwarzschild in ways that matter for accretion
and jets:

  * two horizons at   r_pm = M +/- sqrt(M^2 - a^2)   (they merge at a = M, the
    extremal limit; a > M would be a naked singularity, forbidden);
  * an ERGOSPHERE outside the horizon, r_ergo = M + sqrt(M^2 - a^2 cos^2 theta),
    where spacetime itself is dragged so fast that no observer can stay still --
    the region that powers the Penrose process and Blandford-Znajek jets;
  * a spin-dependent innermost stable circular orbit (ISCO): for PROGRADE orbits
    it shrinks from 6M (a=0) toward 1M (a=M); for RETROGRADE orbits it grows
    toward 9M. Because the ISCO sets the inner edge of the accretion disk and the
    radiative efficiency, measuring it is how black-hole spins are estimated.

This module gives the exact Bardeen-Press-Teukolsky (1972) ISCO formula, the
horizon and ergosphere radii, and the frame-dragging angular velocity. Distances
in units of M. Pure stdlib.
"""

from __future__ import annotations

import math
from typing import Tuple


def horizons(a: float, M: float = 1.0) -> Tuple[float, float]:
    """Outer and inner horizon radii r_pm = M +/- sqrt(M^2 - a^2)."""
    disc = M * M - a * a
    if disc < 0.0:
        raise ValueError("a > M is a naked singularity (forbidden)")
    root = math.sqrt(disc)
    return M + root, M - root


def ergosphere_radius(a: float, theta: float = math.pi / 2, M: float = 1.0) -> float:
    """Static-limit (ergosphere) radius r = M + sqrt(M^2 - a^2 cos^2 theta).
    At the equator (theta = pi/2) it is 2M for any spin."""
    return M + math.sqrt(M * M - a * a * math.cos(theta) ** 2)


def isco_radius(a: float, prograde: bool = True, M: float = 1.0) -> float:
    """Bardeen-Press-Teukolsky ISCO radius for spin a (0<=a<=M).
    r_isco/M = 3 + Z2 -/+ sqrt((3 - Z1)(3 + Z1 + 2 Z2)),
    with the -/+ = prograde/retrograde, and
      Z1 = 1 + (1-a*^2)^{1/3} [ (1+a*)^{1/3} + (1-a*)^{1/3} ],
      Z2 = sqrt(3 a*^2 + Z1^2),   a* = a/M."""
    astar = a / M
    Z1 = 1.0 + (1.0 - astar * astar) ** (1.0 / 3.0) * (
        (1.0 + astar) ** (1.0 / 3.0) + (1.0 - astar) ** (1.0 / 3.0))
    Z2 = math.sqrt(3.0 * astar * astar + Z1 * Z1)
    sign = -1.0 if prograde else 1.0
    r = 3.0 + Z2 + sign * math.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    return r * M


def frame_dragging_omega(r: float, a: float, M: float = 1.0) -> float:
    """Angular velocity of a zero-angular-momentum observer (the frame-dragging
    rate) in the equatorial plane, omega = 2 M a r / (r^4 + ... ). We use the
    equatorial expression omega = 2 M a / (r^3 + a^2 r + 2 M a^2)."""
    return 2.0 * M * a / (r ** 3 + a * a * r + 2.0 * M * a * a)


def horizon_angular_velocity(a: float, M: float = 1.0) -> float:
    """Angular velocity of the outer horizon itself, Omega_H = a / (2 M r_+).
    This is the effective 'spin rate' of the hole that drives the Blandford-
    Znajek mechanism."""
    r_plus, _ = horizons(a, M)
    return a / (2.0 * M * r_plus)
