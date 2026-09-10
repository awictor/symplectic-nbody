"""Sitnikov-problem tests: the route from integrability to chaos.

Claims checked:
  1. For a CIRCULAR binary (e=0) the system is autonomous and conserves energy
     E = 1/2 vz^2 - 1/sqrt(z^2 + r^2) with r = 1/2 fixed.
  2. z = 0 is an exact equilibrium of the on-axis force for all t.
  3. The motion is symmetric: z -> -z flips the acceleration sign (odd force),
     so a reflected initial condition gives the mirror-image orbit.
  4. Sensitivity to initial conditions grows far more violently for an eccentric
     binary than for the integrable circular case -- the onset of chaos.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sitnikov import accel, step_rk4, trajectory, binary_separation  # noqa: E402


def test_circular_conserves_energy():
    ts, zs, vs = trajectory(0.5, 0.0, e=0.0, dt=0.001, n_steps=40000)
    r = 0.5  # constant separation/2 for e=0 (a=1, mu=1)
    E = [0.5 * vs[i] ** 2 - 1.0 / math.sqrt(zs[i] ** 2 + r * r) for i in range(len(zs))]
    drift = max(E) - min(E)
    assert drift < 1e-9, f"e=0 energy not conserved: drift {drift}"


def test_origin_is_equilibrium():
    for t in (0.0, 0.7, 1.9, 3.3):
        for e in (0.0, 0.3):
            assert accel(0.0, t, e) == 0.0, f"z=0 should have zero force at t={t}, e={e}"


def test_force_is_odd_in_z():
    for z in (0.2, 0.9, 1.5):
        for t in (0.0, 1.1):
            assert abs(accel(z, t, 0.2) + accel(-z, t, 0.2)) < 1e-12, "force not odd in z"


def _max_divergence(e, dt=0.002, steps=50000):
    worst = 0.0
    for z0 in (0.3, 0.8, 1.2):
        z1, v1, z2, v2, t = z0, 0.0, z0 + 1e-8, 0.0, 0.0
        for _ in range(steps):
            z1, v1 = step_rk4(z1, v1, t, dt, e)
            z2, v2 = step_rk4(z2, v2, t, dt, e)
            t += dt
            if abs(z1) > 1e4 or abs(z2) > 1e4:
                break
        worst = max(worst, math.sqrt((z1 - z2) ** 2 + (v1 - v2) ** 2) / 1e-8)
    return worst


def test_eccentric_more_chaotic_than_circular():
    circ = _max_divergence(0.0)
    ecc = _max_divergence(0.3)
    assert ecc > 100 * circ, (
        f"eccentric binary should be far more chaotic: e=0 {circ:.1e}, e=0.3 {ecc:.1e}")


def test_separation_oscillates_for_eccentric():
    seps = [binary_separation(t, e=0.5) for t in
            [i * 0.3 for i in range(25)]]
    # e=0.5, a=1 -> separation ranges over [1-e, 1+e] = [0.5, 1.5]
    assert min(seps) < 0.6 and max(seps) > 1.4, f"separation range wrong: {min(seps)}..{max(seps)}"


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
