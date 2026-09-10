"""The magnetic mirror: how converging field lines trap charged particles.

A charged particle spiraling along a magnetic field line conserves its magnetic
moment, the first adiabatic invariant,

    mu = m v_perp^2 / (2 B),

as long as the field changes slowly compared with one gyration. As the particle
drifts into a region of stronger field, mu-conservation forces v_perp^2 to grow, and
since the total speed (kinetic energy) is fixed, the parallel velocity v_par must
shrink. If the field gets strong enough, v_par reaches zero and the particle is
reflected -- a magnetic mirror.

Whether a particle mirrors depends only on its pitch angle alpha (the angle between
its velocity and the field) at the weak-field point. It is trapped if it turns around
before reaching the strong-field throat, which happens when

    sin^2(alpha) > B_min / B_max = 1 / R_m,

where R_m = B_max / B_min is the mirror ratio. Particles with pitch angles inside the
loss cone, sin^2(alpha) < 1/R_m, escape through the throat instead of bouncing.

This is exactly how Earth's dipole field traps the Van Allen radiation belts (bouncing
between the magnetic poles where the field is strongest) and how magnetic-confinement
fusion mirror machines try to hold a plasma. This module gives the magnetic moment,
the loss-cone angle, the trapping verdict, and the mirror-point field, and reproduces
the loss-cone / trapped-particle behaviour. SI units. Pure stdlib; the particle-
trapping companion to the Larmor and Parker-spiral modules.
"""

from __future__ import annotations

import math


def magnetic_moment(m: float, v_perp: float, B: float) -> float:
    """First adiabatic invariant mu = m v_perp^2 / (2 B) (J/T). Conserved as the
    particle moves through a slowly-varying field."""
    return m * v_perp * v_perp / (2.0 * B)


def mirror_ratio(B_max: float, B_min: float) -> float:
    """Mirror ratio R_m = B_max / B_min: how strongly the field converges."""
    return B_max / B_min


def loss_cone_angle(B_max: float, B_min: float) -> float:
    """Loss-cone half-angle (radians): alpha_lc = arcsin(sqrt(B_min/B_max)).
    Particles with pitch angle below this escape; above it, they are trapped."""
    return math.asin(math.sqrt(B_min / B_max))


def is_trapped(pitch_angle_rad: float, B_max: float, B_min: float) -> bool:
    """True if a particle with the given equatorial pitch angle mirrors (is trapped):
    sin^2(alpha) > B_min / B_max."""
    return math.sin(pitch_angle_rad) ** 2 > B_min / B_max


def mirror_field(B_min: float, pitch_angle_rad: float) -> float:
    """Field strength at the mirror point where the particle reflects:
    B_mirror = B_min / sin^2(alpha). If this exceeds B_max the particle is lost."""
    s2 = math.sin(pitch_angle_rad) ** 2
    return B_min / s2 if s2 > 0.0 else float("inf")


def perp_velocity_at(v_total: float, pitch_angle_rad: float, B_min: float,
                     B: float) -> float:
    """Perpendicular velocity (m/s) at field B, from mu-conservation:
    v_perp(B) = v_total sin(alpha) sqrt(B / B_min). Grows toward the mirror point."""
    return v_total * math.sin(pitch_angle_rad) * math.sqrt(B / B_min)
