"""The Parker spiral: why the Sun's magnetic field reaches Earth sideways.

The solar wind drags the Sun's magnetic field radially outward, but the field's
feet stay rooted in a Sun that rotates once every ~25 days. The result is exactly
a rotating-sprinkler pattern: each parcel of plasma flies straight out, yet the
locus of parcels emitted from one footpoint is an Archimedean spiral -- the same
shape water makes leaving a spinning garden hose.

Because the field is frozen into a radial flow of speed u while the source turns
at angular rate Omega, a field line launched from solar radius r0 satisfies

    phi(r) = phi0 - (Omega / u) (r - r0),

and the angle the field makes to the radial direction (the "garden-hose angle") is

    tan(psi) = Omega r sin(theta) / u,

with theta the colatitude (sin theta = 1 in the ecliptic). The winding grows with
distance: near the Sun the field is almost radial, by Earth's orbit psi ~ 45
degrees, and out at Jupiter it is nearly azimuthal. Flux conservation splits the
field into a radial part falling as 1/r^2 and an azimuthal part falling only as
1/r, so far from the Sun the interplanetary field is mostly wound-up azimuthal.

This geometry is why solar energetic particles from a flare reach Earth best when
the flare sits on the Sun's western limb (magnetically connected along the spiral),
and it sets the corotating interaction regions that drive recurrent geomagnetic
storms. This module gives the spiral angle, the field-line shape, and the radial/
azimuthal field split, and reproduces the ~45-degree angle at 1 AU. SI units.
Pure stdlib; the large-scale structure of the Alfvenic solar wind.
"""

from __future__ import annotations

import math

AU = 1.495978707e11
R_SUN = 6.957e8
DAY = 86400.0

# Solar rotation as seen in the wind's inertial frame: the ~25.4-day sidereal
# equatorial period gives Omega ~ 2.86e-6 rad/s.
OMEGA_SUN = 2.0 * math.pi / (25.38 * DAY)


def spiral_angle(r: float, u: float, colat_rad: float = math.pi / 2.0,
                 omega: float = OMEGA_SUN) -> float:
    """Garden-hose angle psi (radians) between the field and the radial direction:
    tan(psi) = Omega r sin(theta) / u."""
    return math.atan(omega * r * math.sin(colat_rad) / u)


def phi_of_r(r: float, u: float, r0: float = R_SUN, phi0: float = 0.0,
             omega: float = OMEGA_SUN) -> float:
    """Azimuth (radians) of a field line at radius r, launched from r0 at phi0:
    phi = phi0 - (Omega/u)(r - r0)."""
    return phi0 - omega / u * (r - r0)


def field_components(r: float, u: float, B0: float, r0: float = AU,
                     colat_rad: float = math.pi / 2.0,
                     omega: float = OMEGA_SUN) -> tuple[float, float]:
    """Radial and azimuthal field (B_r, B_phi) at radius r, normalized so the radial
    field equals B0 at reference radius r0. B_r = B0 (r0/r)^2 (flux conservation);
    B_phi = -B_r (Omega r sin theta / u) (the spiral winding)."""
    B_r = B0 * (r0 / r) ** 2
    B_phi = -B_r * (omega * r * math.sin(colat_rad) / u)
    return B_r, B_phi


def field_magnitude(r: float, u: float, B0: float, r0: float = AU,
                    colat_rad: float = math.pi / 2.0,
                    omega: float = OMEGA_SUN) -> float:
    """Total field strength sqrt(B_r^2 + B_phi^2) at radius r."""
    B_r, B_phi = field_components(r, u, B0, r0, colat_rad, omega)
    return math.hypot(B_r, B_phi)
