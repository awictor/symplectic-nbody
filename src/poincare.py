"""Poincare surface-of-section for the circular restricted three-body problem.

A trajectory in the CR3BP lives on a 3-D energy surface (fixed Jacobi constant)
inside 4-D phase space (x, y, vx, vy). That's hard to see. A Poincare section
cuts it with a plane -- here y = 0 -- and records the point (x, vx) every time
the trajectory crosses the plane in one direction (vy > 0). The 4-D flow becomes
a 2-D map, and its structure jumps out:

  * a regular (quasi-periodic) orbit pierces the plane on a smooth closed curve
    -- an "invariant torus" of KAM theory;
  * a chaotic orbit sprinkles the plane with a diffuse scatter of points.

Same picture, one section: order and chaos coexisting at the same energy. This is
the tool Poincare invented to understand the three-body problem.

Given (x, vx) on the section and a target Jacobi constant C, vy is fixed (up to
sign) by C = 2*Omega - (vx^2 + vy^2), so each section point is a full initial
condition. Pure stdlib; reuses cr3bp.CR3BP.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from cr3bp import CR3BP


def vy_from_jacobi(model: CR3BP, x: float, vx: float, C: float,
                   y: float = 0.0) -> float:
    """Solve C = 2*Omega(x,y) - (vx^2 + vy^2) for vy >= 0. Returns nan if the
    point is energetically forbidden (would need vy^2 < 0)."""
    vy2 = 2.0 * model.effective_potential(x, y) - C - vx * vx
    if vy2 < 0.0:
        return float("nan")
    return math.sqrt(vy2)


def _rk4(f, s, dt):
    k1 = f(s)
    k2 = f([s[i] + 0.5 * dt * k1[i] for i in range(4)])
    k3 = f([s[i] + 0.5 * dt * k2[i] for i in range(4)])
    k4 = f([s[i] + dt * k3[i] for i in range(4)])
    return [s[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]


def section(model: CR3BP, x0: float, vx0: float, C: float,
            n_crossings: int = 200, dt: float = 0.005,
            max_steps: int = 4_000_000) -> List[Tuple[float, float]]:
    """Integrate one orbit on the energy level C and return the (x, vx) points
    where it crosses y = 0 upward (vy > 0)."""
    vy0 = vy_from_jacobi(model, x0, vx0, C)
    if math.isnan(vy0):
        return []
    s = [x0, 0.0, vx0, vy0]
    pts: List[Tuple[float, float]] = []
    prev_y = s[1]
    steps = 0
    while len(pts) < n_crossings and steps < max_steps:
        new = _rk4(model.accel, s, dt)
        steps += 1
        # upward crossing of y = 0: y goes from < 0 to >= 0
        if prev_y < 0.0 <= new[1]:
            # linear interpolation to the crossing for a cleaner point
            frac = -prev_y / (new[1] - prev_y) if (new[1] - prev_y) != 0 else 0.0
            xc = s[0] + frac * (new[0] - s[0])
            vxc = s[2] + frac * (new[2] - s[2])
            pts.append((xc, vxc))
        prev_y = new[1]
        s = new
    return pts


def jacobi_drift(model: CR3BP, x0: float, vx0: float, C: float,
                 steps: int = 20000, dt: float = 0.005) -> float:
    """Max deviation of the Jacobi constant along an orbit -- a fidelity check
    on the section (RK4 is not symplectic, so this bounds the error)."""
    vy0 = vy_from_jacobi(model, x0, vx0, C)
    if math.isnan(vy0):
        return float("nan")
    s = [x0, 0.0, vx0, vy0]
    worst = 0.0
    for _ in range(steps):
        s = _rk4(model.accel, s, dt)
        Cn = model.jacobi_constant(s[0], s[1], s[2], s[3])
        worst = max(worst, abs(Cn - C))
    return worst
