"""The Rossby number and geostrophic balance: why weather spins.

On a rotating planet, any moving parcel of air or water feels the Coriolis force,
which deflects it sideways at a rate set by the Coriolis parameter

    f = 2 Omega sin(latitude),

with Omega the planet's spin rate. Whether rotation matters to a flow is decided by
the dimensionless Rossby number, the ratio of the inertial (advective) acceleration to
the Coriolis acceleration,

    Ro = U / (f L),

for a flow of speed U over length L. When Ro << 1 the Coriolis force dominates and the
flow is *geostrophic*: pressure-gradient and Coriolis forces balance, so the wind
blows along the isobars rather than across them -- the reason cyclones circulate
instead of simply filling in, and highs and lows persist for days. When Ro >> 1 (a
tornado, a bathtub drain, a coffee stir) rotation is negligible and the flow is
governed by inertia and pressure alone.

The natural length that separates the two regimes is the Rossby deformation radius
L_R = c / f (with c a gravity-wave speed), the scale above which rotation reshapes the
flow into balanced vortices -- roughly 1000 km in Earth's atmosphere, setting the size
of weather systems.

This module gives the Coriolis parameter, the Rossby number, the geostrophic-balance
verdict, the geostrophic wind from a pressure gradient, and the deformation radius, and
reproduces the geostrophic synoptic weather / ageostrophic tornado contrast. SI units.
Pure stdlib; the rotating-fluid companion to the atmosphere and Brunt-Vaisala modules.
"""

from __future__ import annotations

import math

OMEGA_EARTH = 7.2921159e-5     # Earth's sidereal spin rate (rad/s)


def coriolis_parameter(latitude_deg: float,
                       omega: float = OMEGA_EARTH) -> float:
    """Coriolis parameter f = 2 Omega sin(lat) (s^-1). Zero at the equator, maximal
    at the poles; negative in the southern hemisphere."""
    return 2.0 * omega * math.sin(math.radians(latitude_deg))


def rossby_number(U: float, L: float, f: float) -> float:
    """Rossby number Ro = U / (f L): inertia vs Coriolis. Ro<<1 rotation-dominated
    (geostrophic); Ro>>1 rotation negligible."""
    return U / (abs(f) * L)


def is_geostrophic(U: float, L: float, f: float, threshold: float = 0.1) -> bool:
    """True if the flow is approximately geostrophic (Ro < threshold): Coriolis and
    pressure gradient balance and the wind follows the isobars."""
    return rossby_number(U, L, f) < threshold


def geostrophic_wind(dp_dn: float, rho: float, f: float) -> float:
    """Geostrophic wind speed (m/s) balancing a pressure gradient dp/dn:
    U_g = (1 / (rho f)) dp/dn. Faster winds accompany tighter isobars."""
    return dp_dn / (rho * abs(f))


def deformation_radius(c: float, f: float) -> float:
    """Rossby deformation radius L_R = c / f (m), with c a gravity-wave speed. The
    scale above which rotation organizes the flow into balanced vortices."""
    return c / abs(f)


def inertial_period(f: float) -> float:
    """Inertial oscillation period 2 pi / f (s): how long a free parcel takes to
    complete one Coriolis-driven loop (half a pendulum day)."""
    return 2.0 * math.pi / abs(f)
