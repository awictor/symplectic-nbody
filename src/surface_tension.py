"""Surface tension: why water climbs, beads, and holds a bug.

A liquid surface behaves like a stretched membrane because molecules at the surface have
fewer neighbours to bond with -- an energy cost per unit area, the surface tension gamma
(N/m, or equivalently J/m^2). It drives three classic effects.

CAPILLARY RISE. In a thin tube water climbs (or mercury falls) to balance the surface
tension pull against gravity, Jurin's law:

    h = 2 gamma cos(theta) / (rho g r),

with theta the contact angle and r the tube radius. Water (theta ~ 0) rises ~1.5 cm in a
1 mm tube and ~15 m in a 1 micron pore -- how sap and groundwater wick upward.

YOUNG-LAPLACE PRESSURE. A curved surface has a pressure jump across it,
delta_P = 2 gamma / r for a droplet (one surface) or 4 gamma / r for a soap bubble (two
surfaces), which is why small bubbles have higher internal pressure than large ones and
the small one empties into the large one when connected.

The same gamma sets the maximum weight a water strider's leg can support and the size of
a drop before it detaches. This module gives the capillary rise, the droplet and bubble
Laplace pressures, and the tube radius for a target rise, and reproduces water's ~1.5 cm
rise and the bubble overpressure. SI units. Pure stdlib; the interface-physics companion
to the Bernoulli and terminal-velocity modules.
"""

from __future__ import annotations

import math

G_EARTH = 9.80665
RHO_WATER = 998.0
GAMMA_WATER = 0.0728          # surface tension of water at 20 C (N/m)
GAMMA_MERCURY = 0.487


def capillary_rise(gamma: float, r: float, rho: float = RHO_WATER,
                   contact_angle_rad: float = 0.0, g: float = G_EARTH) -> float:
    """Jurin's law capillary rise h = 2 gamma cos(theta) / (rho g r) (m). Positive for
    a wetting liquid (theta < 90 deg); negative (depression) for mercury (theta > 90)."""
    return 2.0 * gamma * math.cos(contact_angle_rad) / (rho * g * r)


def droplet_pressure(gamma: float, r: float) -> float:
    """Young-Laplace overpressure inside a liquid droplet: delta_P = 2 gamma / r (Pa).
    One surface."""
    return 2.0 * gamma / r


def bubble_pressure(gamma: float, r: float) -> float:
    """Overpressure inside a soap bubble: delta_P = 4 gamma / r (Pa). Two surfaces (inner
    and outer film), so twice a droplet's."""
    return 4.0 * gamma / r


def radius_for_rise(gamma: float, h: float, rho: float = RHO_WATER,
                    contact_angle_rad: float = 0.0, g: float = G_EARTH) -> float:
    """Tube radius (m) giving a capillary rise h: r = 2 gamma cos(theta) / (rho g h)."""
    return 2.0 * gamma * math.cos(contact_angle_rad) / (rho * g * h)


def surface_energy(gamma: float, area: float) -> float:
    """Surface (free) energy gamma * area (J): the work to create that much new surface."""
    return gamma * area


def max_supported_weight(gamma: float, contact_length: float) -> float:
    """Maximum weight (N) surface tension can hold along a contact line of given length
    (both sides): F = 2 gamma * length -- e.g. a water strider's leg."""
    return 2.0 * gamma * contact_length
