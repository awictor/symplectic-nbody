"""The Ekman spiral: how wind drags a rotating ocean sideways.

When steady wind blows over the sea, friction drives a surface current -- but the Coriolis
force of the rotating Earth deflects it. Balancing friction against Coriolis, Ekman (1905)
found the current doesn't follow the wind: it turns 45 degrees to the right of the wind at
the surface (northern hemisphere), and with depth it spirals clockwise while shrinking
exponentially -- the Ekman spiral.

For eddy viscosity A_z and Coriolis parameter f = 2 Omega sin(latitude), the e-folding
depth (the Ekman depth) is

    D = pi sqrt(2 A_z / |f|),

and the horizontal current at depth z (z <= 0, downward negative) is

    V0 = tau / (rho sqrt(|f| A_z)),          surface speed from wind stress tau
    u(z) = V0 exp(z/d) cos(-pi/4 + z/d),
    v(z) = V0 exp(z/d) sin(-pi/4 + z/d),      d = D/pi = sqrt(2 A_z/|f|),

with x taken along the wind. The depth-integrated transport -- the *Ekman transport* -- is
exactly 90 degrees to the right of the wind, magnitude tau/(rho |f|), independent of the
viscosity: the result that explains coastal upwelling and the piling of water that drives
ocean gyres.

This module gives the Ekman depth, the surface current speed and its 45-degree deflection,
the spiral velocity at any depth, and the net Ekman transport, and reproduces the classic
45-degree surface angle and 90-degree net deflection. SI units. Pure stdlib; the rotating-
fluid boundary-layer companion to the Rossby and geostrophy notes.
"""

from __future__ import annotations

import math

OMEGA_EARTH = 7.2921159e-5     # Earth's rotation rate (rad/s)
RHO_SEAWATER = 1025.0          # kg/m^3


def coriolis_parameter(latitude_deg: float, omega: float = OMEGA_EARTH) -> float:
    """Coriolis parameter f = 2 Omega sin(lat) (1/s). Positive in the north, zero at the
    equator, negative in the south."""
    return 2.0 * omega * math.sin(math.radians(latitude_deg))


def ekman_depth(A_z: float, f: float) -> float:
    """Ekman-layer depth D = pi sqrt(2 A_z / |f|) (m): the current has spiralled through a
    full turn and decayed to e^-pi ~ 4% of the surface value by this depth."""
    return math.pi * math.sqrt(2.0 * A_z / abs(f))


def surface_speed(tau: float, A_z: float, f: float, rho: float = RHO_SEAWATER) -> float:
    """Surface current speed V0 = tau / (rho sqrt(|f| A_z)) (m/s) for wind stress tau (Pa)."""
    return tau / (rho * math.sqrt(abs(f) * A_z))


def surface_angle_deg(f: float) -> float:
    """Signed deflection of the surface current from the wind, measured as a standard math
    angle in (along-wind, cross-wind) coordinates. Northern hemisphere: -45 deg (i.e. 45 deg
    clockwise = to the right of the wind). Southern: +45 deg (to the left)."""
    return -45.0 if f >= 0 else 45.0


def velocity_at_depth(z: float, tau: float, A_z: float, f: float,
                      rho: float = RHO_SEAWATER):
    """Horizontal current (u, v) at depth z (m, negative downward), wind along +x.
    Returns (east/along-wind, north/cross-wind) components in m/s. Northern hemisphere;
    the spiral turns clockwise with depth."""
    V0 = surface_speed(tau, A_z, f, rho)
    d = math.sqrt(2.0 * A_z / abs(f))
    sign = 1.0 if f >= 0 else -1.0
    phase = sign * (-math.pi / 4.0 + z / d)
    decay = math.exp(z / d)
    return V0 * decay * math.cos(phase), V0 * decay * math.sin(phase)


def ekman_transport(tau: float, f: float, rho: float = RHO_SEAWATER) -> float:
    """Depth-integrated Ekman transport magnitude tau/(rho |f|) (m^2/s), directed 90 deg
    to the right of the wind (north) or left (south). Independent of eddy viscosity."""
    return tau / (rho * abs(f))
