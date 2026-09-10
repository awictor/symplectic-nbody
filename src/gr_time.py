"""Gravitational time effects: redshift and the Shapiro delay.

General relativity says clocks run slower deeper in a gravitational well, and
light takes longer to skim past a mass. Two precision tests live here:

GRAVITATIONAL REDSHIFT. A photon climbing out of a potential well loses energy,
so its frequency drops by

    z = Delta nu / nu ~ (Phi_emit - Phi_obs) / c^2 = g h / c^2   (weak field).

The Pound-Rebka experiment (1959) measured this over a 22.5 m tower
(z ~ 2.5e-15); GPS satellites must correct for it (~38 microseconds/day net) or
navigation drifts kilometers.

SHAPIRO DELAY. Radar sent past the Sun and back is delayed by the curved
spacetime near it:

    Delta t = (2 GM/c^3) ln( (r1 + r2 + d)/(r1 + r2 - d) ) ... ,

approximately (4 GM/c^3) ln(4 r1 r2 / b^2) for a grazing ray. The Cassini
measurement (2003) confirmed it to ~1e-5, the tightest Solar-System GR test.

SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.98892e30
M_EARTH = 5.972e24
R_EARTH = 6.371e6
R_SUN = 6.957e8
AU = 1.495978707e11
DAY = 86400.0


def gravitational_redshift(M: float, r_emit: float, r_obs: float) -> float:
    """Fractional frequency shift z = (nu_emit - nu_obs)/nu_obs for a photon
    climbing from r_emit to r_obs (exact Schwarzschild, weak-field reduces to
    Delta Phi / c^2). Positive = redshift (climbing out)."""
    def time_factor(r):
        return math.sqrt(1.0 - 2.0 * G * M / (r * C * C))
    # nu_obs / nu_emit = sqrt(g_tt(emit)/g_tt(obs)); z = nu_emit/nu_obs - 1
    return time_factor(r_obs) / time_factor(r_emit) - 1.0


def redshift_uniform_field(g: float, h: float) -> float:
    """Weak-field redshift over a height h in a uniform field g: z = g h / c^2.
    This is the Pound-Rebka form."""
    return g * h / (C * C)


def gps_time_gain_per_day() -> float:
    """Net special+general relativistic clock gain of a GPS satellite vs the
    ground, in seconds per day. GR (higher potential -> faster clock) dominates
    over SR (orbital speed -> slower clock); the net is ~+38 microseconds/day."""
    r_sat = 26.56e6  # GPS orbital radius (m)
    # GR: fractional rate difference = (Phi_sat - Phi_ground)/c^2 = GM(1/R - 1/r)/c^2
    gr = G * M_EARTH * (1.0 / R_EARTH - 1.0 / r_sat) / (C * C)
    # SR: time dilation from orbital speed v^2 = GM/r (circular), minus Earth-surface
    v_sat2 = G * M_EARTH / r_sat
    # Earth's surface rotation speed is small; approximate ground as ~stationary.
    sr = -0.5 * v_sat2 / (C * C)
    return (gr + sr) * DAY


def shapiro_delay(r1: float, r2: float, b: float, M: float = M_SUN) -> float:
    """Round-trip Shapiro time delay (seconds) for a signal from r1 to r2 passing
    a mass M at impact parameter b (grazing when b ~ R). Uses the standard
    Delta t = (4 GM/c^3) ln(4 r1 r2 / b^2) for a superior-conjunction geometry."""
    return 4.0 * G * M / C ** 3 * math.log(4.0 * r1 * r2 / (b * b))
