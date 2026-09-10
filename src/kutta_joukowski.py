"""Kutta-Joukowski: lift is circulation, and a spinning ball curves.

Why does a wing fly? Not by any "equal transit time" story, but because the flow around it
carries a net circulation -- a swirl -- and a circulating flow in a stream produces a
sideways force. The Kutta-Joukowski theorem makes it exact: the lift per unit span on any
2-D shape is

    L' = rho U Gamma,

the fluid density times the free-stream speed times the circulation Gamma (the line integral
of velocity around the body). The airfoil sets its own circulation through the Kutta
condition -- the flow must leave the sharp trailing edge smoothly -- which for a thin airfoil
at angle of attack alpha gives

    Gamma = pi U c alpha         =>        c_l = 2 pi alpha,

the famous thin-airfoil lift-slope of 2 pi per radian. The same theorem explains the Magnus
effect: a spinning cylinder or ball drags a boundary layer around with it, generating
circulation Gamma = 2 pi r^2 omega (for a cylinder) and a force perpendicular to its motion --
the curve of a topspin tennis ball or a sliced golf shot.

Lift is not free: the circulation that makes it trails vortices, and finite wings pay an
induced drag c_di = c_l^2 / (pi AR e) that falls as the aspect ratio AR grows, which is why
gliders and albatrosses have long thin wings.

This module gives the Kutta-Joukowski lift, the thin-airfoil circulation and lift
coefficient, the Magnus circulation and force, the total lift force from a coefficient, and
the induced drag, and reproduces the 2 pi lift-slope and the Magnus curve of a spinning ball.
SI units, angles in radians unless _deg. Pure stdlib; the aerodynamics companion to the
Bernoulli, Blasius and Strouhal notes.
"""

from __future__ import annotations

import math

RHO_AIR = 1.225


def kutta_joukowski_lift(circulation: float, velocity: float, rho: float = RHO_AIR) -> float:
    """Lift per unit span L' = rho U Gamma (N/m) from the Kutta-Joukowski theorem."""
    return rho * velocity * circulation


def thin_airfoil_circulation(velocity: float, chord: float, alpha: float) -> float:
    """Circulation Gamma = pi U c alpha set by the Kutta condition on a thin airfoil of chord
    c at angle of attack alpha (rad)."""
    return math.pi * velocity * chord * alpha


def lift_coefficient(alpha: float) -> float:
    """Thin-airfoil lift coefficient c_l = 2 pi alpha (alpha in rad): the 2 pi-per-radian
    lift-slope, ~0.11 per degree."""
    return 2.0 * math.pi * alpha


def lift_force(cl: float, velocity: float, area: float, rho: float = RHO_AIR) -> float:
    """Total lift L = c_l * (1/2 rho U^2) * A (N) from a lift coefficient and wing area."""
    return cl * 0.5 * rho * velocity * velocity * area


def magnus_circulation(radius: float, spin_rate: float) -> float:
    """Circulation Gamma = 2 pi r^2 omega around a cylinder spinning at angular rate omega
    (rad/s), assuming the surface drags the fluid with it."""
    return 2.0 * math.pi * radius * radius * spin_rate


def magnus_force(radius: float, spin_rate: float, velocity: float, length: float,
                 rho: float = RHO_AIR) -> float:
    """Magnus (side) force (N) on a spinning cylinder of radius r and length L moving at
    speed U: F = rho U Gamma L with Gamma = 2 pi r^2 omega. The curve of a spinning ball."""
    return kutta_joukowski_lift(magnus_circulation(radius, spin_rate), velocity, rho) * length


def induced_drag_coefficient(cl: float, aspect_ratio: float,
                             efficiency: float = 1.0) -> float:
    """Induced (lift-dependent) drag coefficient c_di = c_l^2 / (pi AR e): the price of lift,
    falling with aspect ratio AR (e is the span efficiency, <=1). Long thin wings pay less."""
    return cl * cl / (math.pi * aspect_ratio * efficiency)
