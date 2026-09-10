"""Virial-theorem tests for a self-gravitating cluster.

Claims checked:
  1. A Plummer sphere is generated near virial equilibrium: its time-averaged
     2T/U sits close to -1.
  2. A COLD cluster starts far from equilibrium (|2T/U| << 1) but relaxes toward
     the virial value -1 as it collapses and mixes (violent relaxation).
  3. Scaling all velocities changes the instantaneous virial ratio by exactly the
     square of the scale factor (2T scales as v^2).
  4. Total energy is still conserved throughout (symplectic integrator).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from virial import (run_virial, virial_ratio, cold_collapse,  # noqa: E402
                    scale_velocities)
from systems import plummer_sphere  # noqa: E402


def test_plummer_is_near_virial():
    s = plummer_sphere(n=200, seed=3)
    _ts, _r, running = run_virial(s, dt=0.02, steps=4000, sample_every=20)
    assert abs(running[-1] - (-1.0)) < 0.15, f"equilibrium 2T/U {running[-1]} not ~ -1"


def test_cold_cluster_relaxes_toward_virial():
    c = cold_collapse(n=200, seed=3, coldness=0.3)
    start = virial_ratio(c)
    _ts, _r, running = run_virial(c, dt=0.02, steps=4000, sample_every=20)
    # starts far from -1 (sub-virial), ends much closer
    assert abs(start - (-1.0)) > 0.5, f"cold start should be far from virial: {start}"
    assert abs(running[-1] - (-1.0)) < 0.2, f"did not relax to virial: {running[-1]}"


def test_velocity_scaling_scales_kinetic():
    s = plummer_sphere(n=100, seed=7)
    T0 = s.kinetic_energy()
    scale_velocities(s, 2.0)
    T1 = s.kinetic_energy()
    assert abs(T1 - 4.0 * T0) < 1e-9, f"kinetic energy should quadruple, {T1} vs {4*T0}"


def test_energy_conserved_during_relaxation():
    c = cold_collapse(n=150, seed=5, coldness=0.4)
    e0 = c.total_energy()
    worst = 0.0
    for s in range(3000):
        c.step("verlet", 0.01)
        if s % 100 == 0:
            worst = max(worst, abs((c.total_energy() - e0) / e0))
    assert worst < 0.05, f"energy drifted too much during collapse: {worst}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
