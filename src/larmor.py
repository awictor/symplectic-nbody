"""The Larmor formula: why accelerating charges radiate.

Maxwell's equations say an accelerating charge emits electromagnetic radiation.
The (non-relativistic) Larmor formula gives the total radiated power:

    P = q^2 a^2 / (6 pi eps0 c^3),

quadratic in the acceleration. It is the root of every classical radiation
process -- synchrotron (magnetic acceleration), bremsstrahlung (Coulomb
deflection), Thomson scattering, and the classical "why doesn't the atom
collapse?" puzzle that demanded quantum mechanics.

Relativistically, for acceleration PERPENDICULAR to the velocity (as in circular
motion) the power is boosted by gamma^4, and for parallel acceleration by
gamma^6:

    P_perp = gamma^4 P_Larmor,   P_par = gamma^6 P_Larmor.

This module gives the Larmor power, the relativistic generalizations, and the
classical-atom collapse time, and reproduces the expected scalings. SI units.
Pure stdlib; it is the physics underneath the synchrotron module.
"""

from __future__ import annotations

import math

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
C = 2.99792458e8
M_E = 9.1093837e-31


def larmor_power(a: float, q: float = E_CHARGE) -> float:
    """Non-relativistic Larmor power P = q^2 a^2 / (6 pi eps0 c^3) (W)."""
    return q * q * a * a / (6.0 * math.pi * EPS0 * C ** 3)


def relativistic_power_perpendicular(a: float, gamma: float,
                                     q: float = E_CHARGE) -> float:
    """Radiated power for acceleration perpendicular to v (circular motion):
    P = gamma^4 P_Larmor."""
    return gamma ** 4 * larmor_power(a, q)


def relativistic_power_parallel(a: float, gamma: float,
                                q: float = E_CHARGE) -> float:
    """Radiated power for acceleration parallel to v (linear accelerator):
    P = gamma^6 P_Larmor."""
    return gamma ** 6 * larmor_power(a, q)


def atom_collapse_time(r0: float = 5.29e-11) -> float:
    """Classical-electromagnetic lifetime of a hydrogen atom: an orbiting
    electron radiating by Larmor spirals into the proton in ~1.6e-11 s -- the
    catastrophe that classical physics could not avoid and quantum mechanics
    resolved. Uses the standard result t = r0^3 / (4 r_e^2 c), r_e the classical
    electron radius."""
    r_e = E_CHARGE ** 2 / (4.0 * math.pi * EPS0 * M_E * C ** 2)  # classical e- radius
    return r0 ** 3 / (4.0 * r_e ** 2 * C)
