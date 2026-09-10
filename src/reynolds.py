"""The Reynolds number: laminar order versus turbulent chaos.

Whether a flow is smooth (laminar) or chaotic (turbulent) is decided by one
dimensionless ratio of inertial to viscous forces,

    Re = rho v L / mu = v L / nu,

with rho the density, v the speed, L a length scale (pipe diameter), mu the dynamic and
nu = mu/rho the kinematic viscosity. Below a critical Re the viscosity damps
disturbances and the flow stays orderly; above it inertia wins and the flow breaks into
eddies. For pipe flow the transition sits near Re ~ 2300 (laminar below, turbulent above
~4000, transitional between).

The regime changes everything. Laminar pipe flow follows the Hagen-Poiseuille law, with
flow rate ~ pressure gradient times radius^4, and a pressure drop linear in velocity;
turbulent flow mixes far better but its drag rises roughly as velocity squared. Re also
governs why a bacterium (Re ~ 1e-5) lives in a world of pure viscosity where coasting is
impossible, while a whale (Re ~ 1e8) glides on inertia.

This module gives the Reynolds number (from properties or kinematic viscosity), the
flow-regime verdict, the Hagen-Poiseuille laminar flow rate, and the entrance length,
and reproduces the ~2300 pipe transition and the microorganism/whale contrast. SI units.
Pure stdlib; the fluid-flow companion to the terminal-velocity and Rossby modules.
"""

from __future__ import annotations

import math

# water at 20 C
RHO_WATER = 998.0
MU_WATER = 1.002e-3            # dynamic viscosity (Pa s)
NU_WATER = MU_WATER / RHO_WATER

RE_LAMINAR = 2300.0           # below this, pipe flow is laminar
RE_TURBULENT = 4000.0        # above this, fully turbulent


def reynolds_number(rho: float, v: float, L: float, mu: float) -> float:
    """Reynolds number Re = rho v L / mu (dimensionless)."""
    return rho * v * L / mu


def reynolds_kinematic(v: float, L: float, nu: float) -> float:
    """Reynolds number from kinematic viscosity: Re = v L / nu."""
    return v * L / nu


def flow_regime(Re: float) -> str:
    """Classify pipe flow: 'laminar' (Re < 2300), 'transitional', or 'turbulent'
    (Re > 4000)."""
    if Re < RE_LAMINAR:
        return "laminar"
    if Re > RE_TURBULENT:
        return "turbulent"
    return "transitional"


def is_turbulent(Re: float) -> bool:
    """True if the flow is (fully) turbulent, Re > 4000."""
    return Re > RE_TURBULENT


def hagen_poiseuille_flow(dP: float, radius: float, length: float,
                          mu: float) -> float:
    """Laminar volumetric flow rate through a pipe (m^3/s):
    Q = pi radius^4 dP / (8 mu length). The r^4 dependence makes narrow pipes throttle
    flow drastically."""
    return math.pi * radius ** 4 * dP / (8.0 * mu * length)


def critical_velocity(L: float, rho: float, mu: float,
                      Re_c: float = RE_LAMINAR) -> float:
    """Speed at which pipe flow of diameter L reaches the critical Reynolds number:
    v = Re_c mu / (rho L)."""
    return Re_c * mu / (rho * L)


def entrance_length(Re: float, D: float) -> float:
    """Laminar hydrodynamic entrance length (m) before the flow is fully developed:
    L_e ~ 0.05 Re D."""
    return 0.05 * Re * D
