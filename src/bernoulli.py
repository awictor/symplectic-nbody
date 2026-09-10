"""Bernoulli's principle: fast flow is low pressure.

Along a streamline of an ideal (incompressible, inviscid) fluid, energy conservation
gives Bernoulli's equation,

    P + (1/2) rho v^2 + rho g h = const,

a trade among static pressure, dynamic pressure (1/2 rho v^2), and gravitational head.
Speed up the flow and the static pressure must drop to keep the sum fixed -- the counter-
intuitive fact behind a lot of everyday physics.

A Venturi (a pipe that narrows) speeds the fluid where it is thin (mass conservation,
A1 v1 = A2 v2) and so lowers the pressure there, which is how a carburettor draws fuel
and a Venturi meter reads flow rate from a pressure difference. A Pitot tube reverses it:
it brings the flow to rest, converting dynamic pressure back to a measurable stagnation
pressure and giving an aircraft's airspeed,

    v = sqrt(2 (P_stagnation - P_static) / rho).

Torricelli's law -- water jetting from a hole at v = sqrt(2 g h) -- is the same equation
with the pressure terms cancelling. (Real airfoil lift needs circulation too, but the
pressure-velocity link is Bernoulli.)

This module gives the Bernoulli constant, the pressure or velocity at a second point,
the Venturi throat velocity from a pressure drop, the Pitot airspeed, and Torricelli's
efflux speed, and reproduces the Pitot airspeed and the Venturi pressure drop. SI units.
Pure stdlib; the streamline companion to the Reynolds and atmosphere modules.
"""

from __future__ import annotations

import math

RHO_WATER = 998.0
RHO_AIR = 1.225
G_EARTH = 9.80665


def bernoulli_constant(P: float, v: float, h: float, rho: float,
                       g: float = G_EARTH) -> float:
    """The conserved head P + (1/2) rho v^2 + rho g h (Pa) along a streamline."""
    return P + 0.5 * rho * v * v + rho * g * h


def pressure_at(P1: float, v1: float, v2: float, rho: float,
                h1: float = 0.0, h2: float = 0.0, g: float = G_EARTH) -> float:
    """Static pressure at point 2 from Bernoulli: P2 = P1 + 1/2 rho (v1^2 - v2^2)
    + rho g (h1 - h2). Faster flow (v2>v1) lowers P2."""
    return P1 + 0.5 * rho * (v1 * v1 - v2 * v2) + rho * g * (h1 - h2)


def venturi_velocity(A1: float, A2: float, dP: float, rho: float) -> float:
    """Throat velocity v2 in a Venturi from the pressure drop dP = P1 - P2 and the
    areas (A1 wide, A2 throat), using continuity A1 v1 = A2 v2:
    v2 = sqrt(2 dP / (rho (1 - (A2/A1)^2)))."""
    ratio = (A2 / A1) ** 2
    return math.sqrt(2.0 * dP / (rho * (1.0 - ratio)))


def pitot_airspeed(P_stagnation: float, P_static: float, rho: float) -> float:
    """Airspeed from a Pitot tube: v = sqrt(2 (P_stag - P_static)/rho). The flow is
    brought to rest, converting dynamic pressure to a measurable difference."""
    return math.sqrt(2.0 * (P_stagnation - P_static) / rho)


def torricelli_speed(h: float, g: float = G_EARTH) -> float:
    """Efflux speed from a hole a depth h below the surface: v = sqrt(2 g h) -- the same
    as free-fall from that height (Bernoulli with equal pressures)."""
    return math.sqrt(2.0 * g * h)


def dynamic_pressure(v: float, rho: float) -> float:
    """Dynamic pressure (1/2) rho v^2 (Pa): the pressure the moving fluid would exert if
    brought to rest."""
    return 0.5 * rho * v * v
