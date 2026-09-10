"""Largest Lyapunov exponent of an N-body system: how fast chaos amplifies.

Two nearby trajectories in a chaotic system separate exponentially:
    |delta(t)| ~ |delta(0)| e^{lambda t}
The largest Lyapunov exponent lambda is the rate. lambda > 0 is the definition of
deterministic chaos: tiny uncertainties blow up, so long-term prediction is
impossible even though the equations are exact.

We estimate lambda with the standard Benettin shadow-trajectory method: evolve a
reference orbit and a twin started an infinitesimal distance away; every few
steps, measure how much the separation grew, accumulate log(growth), and
renormalize the twin back to the original tiny distance so it never saturates.
    lambda ~ (1/T) * sum log(|delta_i| / d0)

Pure stdlib; reuses the project's NBody dynamics and symplectic integrator.
"""

from __future__ import annotations

import math
from typing import List

from nbody import NBody

Vec = List[float]


def _clone(system: NBody) -> NBody:
    return NBody(masses=list(system.m),
                 pos=[list(p) for p in system.pos],
                 vel=[list(v) for v in system.vel],
                 G=system.G, softening=math.sqrt(system.soft2))


def _phase_distance(a: NBody, b: NBody) -> float:
    """Euclidean distance in the full position+velocity phase space."""
    s = 0.0
    for i in range(a.n):
        for k in range(3):
            dp = a.pos[i][k] - b.pos[i][k]
            dv = a.vel[i][k] - b.vel[i][k]
            s += dp * dp + dv * dv
    return math.sqrt(s)


def _rescale(ref: NBody, twin: NBody, d0: float, dist: float) -> None:
    """Pull the twin back toward the reference so the separation is exactly d0,
    keeping the direction of the perturbation."""
    f = d0 / dist
    for i in range(ref.n):
        for k in range(3):
            twin.pos[i][k] = ref.pos[i][k] + (twin.pos[i][k] - ref.pos[i][k]) * f
            twin.vel[i][k] = ref.vel[i][k] + (twin.vel[i][k] - ref.vel[i][k]) * f


def largest_lyapunov(system: NBody, method: str = "forest_ruth",
                     dt: float = 0.001, total_time: float = 200.0,
                     d0: float = 1e-9, renorm_every: int = 50) -> float:
    """Estimate the largest Lyapunov exponent (per unit time)."""
    ref = _clone(system)
    twin = _clone(system)
    # perturb the twin by d0 along the first body's x position
    twin.pos[0][0] += d0

    steps = int(total_time / dt)
    log_sum = 0.0
    t_elapsed = 0.0
    for s in range(steps):
        ref.step(method, dt)
        twin.step(method, dt)
        t_elapsed += dt
        if (s + 1) % renorm_every == 0:
            dist = _phase_distance(ref, twin)
            if dist > 0.0:
                log_sum += math.log(dist / d0)
                _rescale(ref, twin, d0, dist)
    return log_sum / t_elapsed


def lyapunov_time(system: NBody, **kwargs) -> float:
    """The e-folding time 1/lambda -- the horizon beyond which prediction fails.
    Returns +inf for regular (non-chaotic) motion."""
    lam = largest_lyapunov(system, **kwargs)
    if lam <= 1e-6:
        return float("inf")
    return 1.0 / lam
