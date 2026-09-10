"""The Blasius boundary layer: the thin sheared film on a flat plate.

A real fluid sticks to a surface -- the no-slip condition -- so a plate held in a stream has
a thin layer near it where the velocity climbs from zero at the wall to the free-stream value
U. Prandtl saw that at high Reynolds number this layer is thin and can be treated separately
from the inviscid outer flow; Blasius (1908) solved the resulting equation for a flat plate
exactly (as a similarity ODE), and the answers are clean power laws in the distance x from
the leading edge.

The local Reynolds number is Re_x = U x / nu. The boundary-layer thickness (where the speed
reaches 99% of U) grows as

    delta(x) = 5.0 x / sqrt(Re_x),

so it thickens as sqrt(x) -- millimetres over the front of a wing. Two more measures come out
of the same profile: the displacement thickness delta* = 1.721 x/sqrt(Re_x) (how far the wall
effectively pushes the outer streamlines out) and the momentum thickness
theta = 0.664 x/sqrt(Re_x). The wall shear sets the local skin-friction coefficient

    c_f = 0.664 / sqrt(Re_x),

and integrating it over a plate of length L gives the total drag coefficient
C_D = 1.328 / sqrt(Re_L). The layer stays laminar until Re_x ~ 5e5, where transition to
turbulence begins.

This module gives the local Reynolds number, the three thicknesses, the local and average
skin-friction coefficients, the drag force, and the laminar/turbulent transition point, and
reproduces the 5.0/sqrt(Re) thickness law and the 1.328/sqrt(Re_L) plate drag. SI units.
Pure stdlib; the viscous-flow companion to the Reynolds, convection and nozzle notes.
"""

from __future__ import annotations

import math

RE_TRANSITION = 5.0e5          # flat-plate laminar-to-turbulent transition


def reynolds_x(velocity: float, x: float, nu: float) -> float:
    """Local Reynolds number Re_x = U x / nu at distance x from the leading edge, for a
    free-stream speed U and kinematic viscosity nu."""
    return velocity * x / nu


def bl_thickness(velocity: float, x: float, nu: float) -> float:
    """Boundary-layer (99%) thickness delta = 5.0 x / sqrt(Re_x) (m). Grows as sqrt(x)."""
    return 5.0 * x / math.sqrt(reynolds_x(velocity, x, nu))


def displacement_thickness(velocity: float, x: float, nu: float) -> float:
    """Displacement thickness delta* = 1.721 x / sqrt(Re_x) (m): the outward shift of the
    outer streamlines caused by the slowed near-wall fluid."""
    return 1.721 * x / math.sqrt(reynolds_x(velocity, x, nu))


def momentum_thickness(velocity: float, x: float, nu: float) -> float:
    """Momentum thickness theta = 0.664 x / sqrt(Re_x) (m): the momentum deficit of the
    boundary layer, the length that governs drag."""
    return 0.664 * x / math.sqrt(reynolds_x(velocity, x, nu))


def skin_friction_local(velocity: float, x: float, nu: float) -> float:
    """Local skin-friction coefficient c_f = 0.664 / sqrt(Re_x): the wall shear stress
    normalized by the dynamic pressure, at distance x."""
    return 0.664 / math.sqrt(reynolds_x(velocity, x, nu))


def skin_friction_average(velocity: float, length: float, nu: float) -> float:
    """Plate-averaged (total) drag coefficient C_D = 1.328 / sqrt(Re_L) over a plate of
    length L -- exactly twice the local c_f at the trailing edge."""
    return 1.328 / math.sqrt(reynolds_x(velocity, length, nu))


def drag_force(velocity: float, length: float, width: float, nu: float,
               rho: float) -> float:
    """Total friction drag (N) on one side of a flat plate of length L and width w:
    F = C_D * (1/2 rho U^2) * (L w)."""
    cd = skin_friction_average(velocity, length, nu)
    return cd * 0.5 * rho * velocity * velocity * length * width


def transition_distance(velocity: float, nu: float,
                        re_transition: float = RE_TRANSITION) -> float:
    """Distance from the leading edge (m) at which the laminar boundary layer transitions to
    turbulence, where Re_x reaches ~5e5: x = Re_transition nu / U."""
    return re_transition * nu / velocity
