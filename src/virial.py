"""The virial theorem and violent relaxation of a self-gravitating cluster.

For a bound gravitational system in equilibrium the time-averaged kinetic and
potential energies obey the virial theorem:

    2 <T> + <U> = 0        equivalently   <T> / |<U>| = 1/2,   2<T>/<U> = -1.

A system that starts far from this balance (e.g. a "cold" cluster with too little
kinetic energy) collapses, overshoots, and settles -- "violent relaxation" -- to a
new equilibrium that DOES satisfy the virial theorem. This module measures the
running virial ratio along an N-body integration and provides a cold-collapse
setup to watch relaxation happen.

Reuses NBody + the symplectic integrator + the Plummer generator. Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from nbody import NBody
from systems import plummer_sphere


def virial_ratio(system: NBody) -> float:
    """Instantaneous 2T/U. Equals -1 for a system exactly in virial balance."""
    T = system.kinetic_energy()
    U = system.potential_energy()
    if U == 0.0:
        return float("nan")
    return 2.0 * T / U


def scale_velocities(system: NBody, factor: float) -> None:
    """Multiply every velocity by `factor` (used to make a cluster hotter/colder
    than virial equilibrium). factor<1 => cold => the cluster will collapse."""
    for i in range(system.n):
        for k in range(3):
            system.vel[i][k] *= factor


def run_virial(system: NBody, dt: float, steps: int, sample_every: int = 1,
               warmup_frac: float = 0.3):
    """Integrate and return (times, virial_ratios, running_time_average).

    The running average is taken only over samples after `warmup_frac` of the run
    so that transient relaxation doesn't bias the equilibrium estimate."""
    ts, ratios, running = [], [], []
    acc_sum, acc_n = 0.0, 0
    start_avg = int(steps * warmup_frac)
    t = 0.0
    for s in range(steps):
        system.step("verlet", dt)
        t += dt
        if s % sample_every == 0:
            r = virial_ratio(system)
            ts.append(t)
            ratios.append(r)
            if s >= start_avg:
                acc_sum += r
                acc_n += 1
            running.append(acc_sum / acc_n if acc_n else r)
    return ts, ratios, running


def cold_collapse(n: int = 200, seed: int = 3, coldness: float = 0.3) -> NBody:
    """A Plummer sphere started 'cold' (velocities scaled down by `coldness`), so
    it is sub-virial and will collapse then relax toward equilibrium."""
    sys_ = plummer_sphere(n=n, seed=seed)
    scale_velocities(sys_, coldness)
    return sys_
