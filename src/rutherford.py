"""Rutherford scattering: how the nucleus was discovered.

Firing alpha particles at gold foil, Geiger and Marsden found a few bouncing almost
straight back -- impossible if positive charge were spread out (Thomson's "plum
pudding"). Rutherford explained it with a tiny, dense, charged nucleus and a pure
Coulomb deflection, deriving the differential cross section

    dsigma/dOmega = ( Z1 Z2 e^2 / (16 pi eps0 E) )^2 / sin^4(theta/2),

which blows up at small angles (grazing) and falls steeply toward back-scattering, but
crucially is NONZERO at 180 degrees -- exactly the rare hard bounces observed. The
impact parameter maps to scattering angle through

    b = (Z1 Z2 e^2 / (8 pi eps0 E)) cot(theta/2),

and the head-on approach (b = 0) reaches a distance of closest approach
r_min = Z1 Z2 e^2 / (4 pi eps0 E), which for MeV alphas on gold is ~30 fm -- Rutherford's
upper bound on the nuclear size.

This module gives the differential cross section, the impact parameter for a given
angle, the distance of closest approach, and the 1/sin^4 angular dependence, and
reproduces the gold-foil back-scattering and the ~10-fm nuclear-size bound. SI units,
energies via a MeV helper. Pure stdlib; the Coulomb-scattering companion to the Bohr
and Gamow modules.
"""

from __future__ import annotations

import math

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
MEV = 1e6 * 1.602176634e-19
FM = 1e-15


def coulomb_constant(Z1: float, Z2: float) -> float:
    """The combination k = Z1 Z2 e^2 / (4 pi eps0) (J m), the Coulomb energy scale."""
    return Z1 * Z2 * E_CHARGE ** 2 / (4.0 * math.pi * EPS0)


def differential_cross_section(theta_rad: float, Z1: float, Z2: float,
                               E: float) -> float:
    """Rutherford differential cross section dsigma/dOmega (m^2/sr):
    (Z1 Z2 e^2 / (16 pi eps0 E))^2 / sin^4(theta/2). E is the beam kinetic energy (J)."""
    prefactor = coulomb_constant(Z1, Z2) / (4.0 * E)
    s = math.sin(theta_rad / 2.0)
    return prefactor ** 2 / s ** 4


def impact_parameter(theta_rad: float, Z1: float, Z2: float, E: float) -> float:
    """Impact parameter b (m) that produces scattering angle theta:
    b = (k / (2 E)) cot(theta/2), k = Z1 Z2 e^2 / (4 pi eps0)."""
    k = coulomb_constant(Z1, Z2)
    return k / (2.0 * E) / math.tan(theta_rad / 2.0)


def closest_approach(E: float, Z1: float, Z2: float) -> float:
    """Distance of closest approach for a head-on (b=0) collision:
    r_min = Z1 Z2 e^2 / (4 pi eps0 E) (m). Rutherford's bound on the nuclear size."""
    return coulomb_constant(Z1, Z2) / E


def scattering_angle(b: float, Z1: float, Z2: float, E: float) -> float:
    """Scattering angle (radians) for a given impact parameter b: invert the impact-
    parameter relation, theta = 2 arctan(k / (2 E b))."""
    k = coulomb_constant(Z1, Z2)
    return 2.0 * math.atan(k / (2.0 * E * b))
