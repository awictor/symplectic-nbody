"""Fourth-order Hermite predictor-corrector integrator.

This is the workhorse of modern collisional stellar dynamics (star clusters,
galactic nuclei). Unlike Runge-Kutta, it needs only ONE force evaluation per step
by also computing the analytic jerk (da/dt) and using a Hermite interpolation of
acceleration + jerk to reach fourth order.

Per step (Makino & Aarseth 1992):
  1. predict positions/velocities to O(dt^3) using current a, jdot;
  2. evaluate a, jdot at the predicted state (the single force call);
  3. build the 2nd/3rd acceleration derivatives from the Hermite interpolant;
  4. correct positions/velocities to fourth order.

The scheme is time-symmetric-ish and famously well-behaved for close encounters.
Pure stdlib; the jerk is computed exactly from the pairwise gravity.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

Vec = List[float]
AccelJerk = Callable[[List[Vec], List[Vec]], Tuple[List[Vec], List[Vec]]]


def accel_and_jerk(pos: List[Vec], vel: List[Vec], masses: List[float],
                   G: float = 1.0, soft2: float = 0.0) -> Tuple[List[Vec], List[Vec]]:
    """Exact Newtonian acceleration a_i and jerk (da/dt)_i for every body.

    jerk_i = G sum_j m_j [ v_ij / r^3 - 3 (r_ij . v_ij) r_ij / r^5 ],
    with r_ij = r_j - r_i, v_ij = v_j - v_i.
    """
    n = len(pos)
    a = [[0.0, 0.0, 0.0] for _ in range(n)]
    j = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(n):
        for k in range(i + 1, n):
            dx = pos[k][0] - pos[i][0]
            dy = pos[k][1] - pos[i][1]
            dz = pos[k][2] - pos[i][2]
            dvx = vel[k][0] - vel[i][0]
            dvy = vel[k][1] - vel[i][1]
            dvz = vel[k][2] - vel[i][2]
            r2 = dx * dx + dy * dy + dz * dz + soft2
            r = math.sqrt(r2)
            inv_r3 = 1.0 / (r2 * r)
            inv_r5 = inv_r3 / r2
            rv = dx * dvx + dy * dvy + dz * dvz
            # acceleration on i from k, and equal/opposite on k from i
            gmk = G * masses[k]
            gmi = G * masses[i]
            a[i][0] += gmk * inv_r3 * dx; a[i][1] += gmk * inv_r3 * dy; a[i][2] += gmk * inv_r3 * dz
            a[k][0] -= gmi * inv_r3 * dx; a[k][1] -= gmi * inv_r3 * dy; a[k][2] -= gmi * inv_r3 * dz
            # jerk
            jx = inv_r3 * dvx - 3.0 * rv * inv_r5 * dx
            jy = inv_r3 * dvy - 3.0 * rv * inv_r5 * dy
            jz = inv_r3 * dvz - 3.0 * rv * inv_r5 * dz
            j[i][0] += gmk * jx; j[i][1] += gmk * jy; j[i][2] += gmk * jz
            j[k][0] -= gmi * jx; j[k][1] -= gmi * jy; j[k][2] -= gmi * jz
    return a, j


def hermite_step(pos: List[Vec], vel: List[Vec], masses: List[float],
                 dt: float, G: float = 1.0, soft2: float = 0.0
                 ) -> Tuple[List[Vec], List[Vec]]:
    """One 4th-order Hermite predictor-corrector step. Returns (pos, vel)."""
    n = len(pos)
    a0, j0 = accel_and_jerk(pos, vel, masses, G, soft2)

    # 1. predict
    pp = [[pos[i][k] + dt * vel[i][k] + dt * dt / 2.0 * a0[i][k]
           + dt ** 3 / 6.0 * j0[i][k] for k in range(3)] for i in range(n)]
    pv = [[vel[i][k] + dt * a0[i][k] + dt * dt / 2.0 * j0[i][k]
           for k in range(3)] for i in range(n)]

    # 2. evaluate at predicted state
    a1, j1 = accel_and_jerk(pp, pv, masses, G, soft2)

    # 3-4. correct (Hermite interpolation to 4th order)
    new_pos = [[0.0, 0.0, 0.0] for _ in range(n)]
    new_vel = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(n):
        for k in range(3):
            # 2nd and 3rd derivatives of acceleration at t (from a,j at both ends)
            a2 = (-6.0 * (a0[i][k] - a1[i][k]) - dt * (4.0 * j0[i][k] + 2.0 * j1[i][k])) / (dt * dt)
            a3 = (12.0 * (a0[i][k] - a1[i][k]) + 6.0 * dt * (j0[i][k] + j1[i][k])) / (dt ** 3)
            new_vel[i][k] = (vel[i][k] + dt * a0[i][k] + dt * dt / 2.0 * j0[i][k]
                             + dt ** 3 / 6.0 * a2 + dt ** 4 / 24.0 * a3)
            new_pos[i][k] = (pos[i][k] + dt * vel[i][k] + dt * dt / 2.0 * a0[i][k]
                             + dt ** 3 / 6.0 * j0[i][k] + dt ** 4 / 24.0 * a2
                             + dt ** 5 / 120.0 * a3)
    return new_pos, new_vel
