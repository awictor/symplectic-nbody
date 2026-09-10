"""Bi-elliptic transfer: when three burns beat two.

The Hohmann transfer -- two burns on a single ellipse tangent to both orbits -- is the
cheapest way between two circular orbits for modest radius changes. But for very large
ratios there is a cheaper, stranger option: the bi-elliptic transfer, which uses THREE
burns and a detour far beyond the target.

  1. burn to an ellipse whose apoapsis is at a large radius r_b (much bigger than the
     target r2);
  2. at r_b, a small burn raising periapsis to r2;
  3. at r2, a retro-burn to circularize.

Counter-intuitively, flinging the spacecraft way out to r_b and back can cost less
total delta-v than the direct Hohmann, because the middle burn happens where orbital
speeds are tiny and the Oberth penalty for plane changes / raising is small.

Which wins depends only on the ratios R = r2/r1 and the detour ratio r_b/r1:

  * R < 11.94  -> Hohmann is always cheaper;
  * R > 15.58  -> bi-elliptic is cheaper for any sufficiently large r_b;
  * 11.94 < R < 15.58 -> depends on r_b.

The famous crossover at R = 11.94 is one of the tidy exact numbers of astrodynamics.
The cost is a small delta-v saving paid for with a much longer transfer time (the
detour can take years), so bi-elliptic is used only for extreme orbit changes.

This module gives the Hohmann and bi-elliptic total delta-v, the decision between them,
and the classic 11.94 crossover, and reproduces the regime where bi-elliptic wins. SI
units. Pure stdlib; the multi-burn companion to the Hohmann and Oberth modules.
"""

from __future__ import annotations

import math

MU_EARTH = 3.986004418e14      # Earth's gravitational parameter (m^3/s^2)

CROSSOVER_LOW = 11.93876       # R below which Hohmann always wins
CROSSOVER_HIGH = 15.58175      # R above which bi-elliptic always wins (large r_b)


def _circular(mu: float, r: float) -> float:
    return math.sqrt(mu / r)


def hohmann_delta_v(r1: float, r2: float, mu: float = MU_EARTH) -> float:
    """Total delta-v (m/s) of a two-burn Hohmann transfer between circular orbits
    r1 and r2."""
    v1 = _circular(mu, r1)
    v2 = _circular(mu, r2)
    a_t = 0.5 * (r1 + r2)
    v_p = math.sqrt(mu * (2.0 / r1 - 1.0 / a_t))   # periapsis speed on transfer
    v_a = math.sqrt(mu * (2.0 / r2 - 1.0 / a_t))   # apoapsis speed on transfer
    return abs(v_p - v1) + abs(v2 - v_a)


def bi_elliptic_delta_v(r1: float, r2: float, r_b: float,
                        mu: float = MU_EARTH) -> float:
    """Total delta-v (m/s) of a three-burn bi-elliptic transfer via an intermediate
    apoapsis r_b (> r2)."""
    v1 = _circular(mu, r1)
    v2 = _circular(mu, r2)
    a1 = 0.5 * (r1 + r_b)     # first ellipse: r1 -> r_b
    a2 = 0.5 * (r2 + r_b)     # second ellipse: r_b -> r2
    # burn 1 at r1: circular -> ellipse a1
    dv1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a1)) - v1
    # burn 2 at r_b: ellipse a1 -> ellipse a2
    dv2 = abs(math.sqrt(mu * (2.0 / r_b - 1.0 / a2))
              - math.sqrt(mu * (2.0 / r_b - 1.0 / a1)))
    # burn 3 at r2: ellipse a2 -> circular (retro)
    dv3 = abs(math.sqrt(mu * (2.0 / r2 - 1.0 / a2)) - v2)
    return abs(dv1) + dv2 + dv3


def bi_elliptic_is_cheaper(r1: float, r2: float, r_b: float,
                           mu: float = MU_EARTH) -> bool:
    """True if the bi-elliptic transfer costs less total delta-v than the Hohmann."""
    return bi_elliptic_delta_v(r1, r2, r_b, mu) < hohmann_delta_v(r1, r2, mu)


def crossover_ratio() -> float:
    """The classic radius ratio R = r2/r1 = 11.94 below which Hohmann always wins,
    regardless of the bi-elliptic detour radius."""
    return CROSSOVER_LOW
