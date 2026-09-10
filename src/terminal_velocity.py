"""Terminal velocity and drag: how fast things fall through a fluid.

A body falling through air or water speeds up until drag balances gravity, then coasts
at its terminal velocity. Which drag law applies depends on the Reynolds number
Re = rho v L / mu, the ratio of inertial to viscous forces.

At high Reynolds number (a skydiver, a raindrop, a cannonball) drag is quadratic --
the body must shove aside a column of fluid,

    F_drag = (1/2) C_d rho A v^2,

so balancing weight m g gives

    v_term = sqrt( 2 m g / (C_d rho A) ).

At low Reynolds number (a bacterium, a fog droplet, a dust grain settling in honey)
viscosity dominates and Stokes' law applies, F = 6 pi mu r v, giving a terminal speed
that rises as the square of the radius,

    v_Stokes = 2 r^2 (rho_p - rho_f) g / (9 mu).

That r^2 dependence is why fine dust and cloud droplets fall so slowly they effectively
float, while hailstones plummet. A human skydiver reaches ~55 m/s (120 mph) belly-down;
a large raindrop ~9 m/s; a fog droplet ~1 cm/s.

This module gives the quadratic-drag terminal velocity, the Stokes terminal velocity,
the Reynolds number that selects between them, and the drag force, and reproduces the
skydiver and raindrop speeds. SI units. Pure stdlib; the fluid-resistance companion to
the atmosphere and Rossby modules.
"""

from __future__ import annotations

import math

G_EARTH = 9.80665
RHO_AIR = 1.225                # sea-level air density (kg/m^3)
MU_AIR = 1.81e-5              # dynamic viscosity of air (Pa s)
RHO_WATER = 1000.0


def terminal_velocity(m: float, A: float, C_d: float = 0.47,
                      rho_fluid: float = RHO_AIR, g: float = G_EARTH) -> float:
    """High-Re terminal velocity v = sqrt(2 m g / (C_d rho A)) (m/s). C_d ~ 0.47 for
    a sphere, ~1.0-1.3 for a belly-down skydiver."""
    return math.sqrt(2.0 * m * g / (C_d * rho_fluid * A))


def drag_force(v: float, A: float, C_d: float = 0.47,
               rho_fluid: float = RHO_AIR) -> float:
    """Quadratic drag force F = (1/2) C_d rho A v^2 (N)."""
    return 0.5 * C_d * rho_fluid * A * v * v


def reynolds_number(v: float, L: float, rho_fluid: float = RHO_AIR,
                    mu: float = MU_AIR) -> float:
    """Reynolds number Re = rho v L / mu: inertial vs viscous. Re<<1 Stokes regime,
    Re>>1 quadratic-drag regime."""
    return rho_fluid * v * L / mu


def stokes_velocity(r: float, rho_p: float, rho_f: float = RHO_AIR,
                    mu: float = MU_AIR, g: float = G_EARTH) -> float:
    """Low-Re Stokes terminal velocity v = 2 r^2 (rho_p - rho_f) g / (9 mu) (m/s),
    rising as the square of the particle radius."""
    return 2.0 * r * r * (rho_p - rho_f) * g / (9.0 * mu)


def stokes_drag(r: float, v: float, mu: float = MU_AIR) -> float:
    """Stokes drag force F = 6 pi mu r v (N), linear in velocity."""
    return 6.0 * math.pi * mu * r * v


def sphere_terminal_velocity(r: float, rho_p: float, C_d: float = 0.47,
                             rho_f: float = RHO_AIR, g: float = G_EARTH) -> float:
    """High-Re terminal velocity of a solid sphere of radius r and density rho_p,
    using m = (4/3) pi r^3 rho_p and A = pi r^2:
    v = sqrt(8 r rho_p g / (3 C_d rho_f))."""
    return math.sqrt(8.0 * r * rho_p * g / (3.0 * C_d * rho_f))
