"""The Stefan problem: how fast a melting or freezing front advances.

Freeze a pond from the top, or melt a solid from one face, and a sharp interface -- the
phase-change front -- eats into the material. It is not driven by the temperature alone but
by the latent heat that must be carried away (freezing) or delivered (melting) at the front,
so the front does not move at constant speed: it slows as it advances, going as sqrt(t).

For a half-space held at a fixed surface temperature T_s past the melt point T_m, the front
position is

    X(t) = 2 lambda sqrt(alpha t),

where alpha = k/(rho c_p) is the thermal diffusivity and the dimensionless lambda solves the
transcendental Stefan condition

    lambda exp(lambda^2) erf(lambda) = St / sqrt(pi),      St = c_p (T_s - T_m) / L,

with L the latent heat of fusion and St the Stefan number (sensible heat available divided by
latent heat needed). Small St (latent heat dominates) gives a slow front, lambda ~ sqrt(St/2);
large St gives a fast one. The time to freeze a layer of thickness H then scales as H^2, the
same diffusive law that makes thin ice form fast and thick ice form ever slower -- and it is
the classic estimate for how quickly a lake ices over or a lava lake crusts.

This module gives the Stefan number, solves for lambda, returns the front position and the
time to reach a depth, and reproduces Stefan's ice result (a few cm of ice in a day of hard
frost) and the sqrt(t) advance. SI units. Pure stdlib (erf from math, lambda by bisection);
the moving-boundary companion to the Fick-diffusion and convection notes.
"""

from __future__ import annotations

import math


def stefan_number(specific_heat: float, delta_T: float, latent_heat: float) -> float:
    """Stefan number St = c_p dT / L: sensible heat available per unit latent heat needed.
    dT is the surface overshoot past the phase-change temperature."""
    return specific_heat * delta_T / latent_heat


def solve_lambda(stefan: float) -> float:
    """Solve the Stefan condition lambda exp(lambda^2) erf(lambda) = St/sqrt(pi) for the
    growth coefficient lambda (dimensionless), by bisection. lambda -> sqrt(St/2) for small
    St and grows without bound as St increases."""
    target = stefan / math.sqrt(math.pi)

    def f(lam):
        return lam * math.exp(lam * lam) * math.erf(lam) - target

    lo, hi = 1e-9, 1.0
    while f(hi) < 0.0:          # expand until the root is bracketed
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def thermal_diffusivity(conductivity: float, density: float, specific_heat: float) -> float:
    """Thermal diffusivity alpha = k / (rho c_p) (m^2/s)."""
    return conductivity / (density * specific_heat)


def front_position(t: float, lam: float, alpha: float) -> float:
    """Position of the phase-change front at time t: X = 2 lambda sqrt(alpha t) (m). The
    sqrt(t) advance -- the front slows as it deepens."""
    return 2.0 * lam * math.sqrt(alpha * t)


def front_speed(t: float, lam: float, alpha: float) -> float:
    """Speed of the front dX/dt = lambda sqrt(alpha / t) (m/s): falls off as 1/sqrt(t)."""
    if t <= 0.0:
        return float("inf")
    return lam * math.sqrt(alpha / t)


def time_to_depth(depth: float, lam: float, alpha: float) -> float:
    """Time for the front to reach a given depth: t = (depth / (2 lambda))^2 / alpha (s).
    Scales as depth^2 -- thick layers form far more slowly than thin ones."""
    return (depth / (2.0 * lam)) ** 2 / alpha
