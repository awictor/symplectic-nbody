"""The Casimir effect: two plates pushed together by empty space.

Quantum field theory says the vacuum is not empty -- every electromagnetic mode has a
zero-point energy hbar omega / 2, even with no photons present. Between two parallel
conducting plates only modes whose wavelengths fit in the gap survive, so the vacuum between
the plates holds fewer modes than the vacuum outside. The outside pushes harder, and the
plates are drawn together by a pressure that comes from nothing but the structure of empty
space itself. Casimir predicted it in 1948; it was measured to a few percent in 1997.

For two ideal plates of area A separated by distance d the attractive force and pressure are

    F = pi^2 hbar c A / (240 d^4),      P = pi^2 hbar c / (240 d^4),

the steep d^-4 dependence that makes the effect utterly negligible at macroscopic gaps but
dominant below ~100 nm -- where it causes stiction in micro-electromechanical systems (MEMS)
and must be engineered around. The associated vacuum energy per area is

    E/A = -pi^2 hbar c / (720 d^3).

This module gives the Casimir force, pressure and energy, the gap at which the Casimir
pressure equals a target (e.g. atmospheric), and a comparison to the plates' gravitational
attraction, and reproduces the ~1 Pa pressure at a 100 nm gap and the measured micro-newton-
scale forces. SI units. Pure stdlib; the quantum-vacuum companion to the blackbody and
uncertainty notes.
"""

from __future__ import annotations

HBAR = 1.054571817e-34        # reduced Planck constant (J s)
C = 299792458.0               # speed of light (m/s)
G = 6.674e-11                 # gravitational constant (SI)


def casimir_pressure(separation: float) -> float:
    """Casimir pressure P = pi^2 hbar c / (240 d^4) (Pa) between two ideal parallel plates.
    Attractive; scales as d^-4 so it explodes at nanometre gaps."""
    import math
    return math.pi ** 2 * HBAR * C / (240.0 * separation ** 4)


def casimir_force(area: float, separation: float) -> float:
    """Casimir force F = P * A = pi^2 hbar c A / (240 d^4) (N) drawing the plates together."""
    return casimir_pressure(separation) * area


def casimir_energy_per_area(separation: float) -> float:
    """Casimir vacuum energy per unit area E/A = -pi^2 hbar c / (720 d^3) (J/m^2). Negative:
    the gap lowers the vacuum energy, and the force is its gradient."""
    import math
    return -math.pi ** 2 * HBAR * C / (720.0 * separation ** 3)


def separation_for_pressure(pressure: float) -> float:
    """Plate gap (m) at which the Casimir pressure reaches a target value:
    d = (pi^2 hbar c / (240 P))^(1/4). ~10 nm for atmospheric pressure."""
    import math
    return (math.pi ** 2 * HBAR * C / (240.0 * pressure)) ** 0.25


def gravitational_pressure(separation: float, density: float, thickness: float) -> float:
    """Newtonian gravitational attraction pressure between two slabs of given density and
    thickness at separation d (infinite-plate limit P = 2 pi G rho^2 t^2 ... ), here the
    finite-slab estimate 2 pi G (rho t)^2 for d << plate size. For comparison with Casimir."""
    import math
    sigma = density * thickness       # mass per area
    return 2.0 * math.pi * G * sigma * sigma


def casimir_dominates_gravity(separation: float, density: float, thickness: float) -> bool:
    """True if the Casimir pressure exceeds the gravitational attraction at this gap -- true
    at small separations, since Casimir grows as d^-4 while (thin-slab) gravity is nearly
    flat."""
    return casimir_pressure(separation) > gravitational_pressure(separation, density, thickness)
