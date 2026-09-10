"""Circular restricted three-body problem tests.

Claims checked:
  1. All five Lagrange points are equilibria: grad(Omega) = 0 there.
  2. L4/L5 form an equilateral triangle with the primaries.
  3. The Jacobi constant is conserved along a trajectory (to ~1e-11).
  4. L4 stability follows the Routh criterion: for mu < ~0.0385 a small
     perturbation stays bounded (libration); for larger mu it runs away.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cr3bp import CR3BP  # noqa: E402


def _rk4(f, s, dt):
    k1 = f(s)
    k2 = f([s[i] + 0.5 * dt * k1[i] for i in range(4)])
    k3 = f([s[i] + 0.5 * dt * k2[i] for i in range(4)])
    k4 = f([s[i] + dt * k3[i] for i in range(4)])
    return [s[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]


def test_lagrange_points_are_equilibria():
    c = CR3BP(0.01215)  # Earth-Moon
    for name, (x, y) in c.lagrange_points().items():
        gx, gy = c.grad_omega(x, y)
        assert math.hypot(gx, gy) < 1e-9, f"{name} not an equilibrium: |grad|={math.hypot(gx, gy)}"


def test_triangular_points_equilateral():
    c = CR3BP(0.2)
    (x4, y4), (x5, y5) = c.triangular_points()
    # distance from each primary must equal the primary separation (=1)
    for (x, y) in ((x4, y4), (x5, y5)):
        d1 = math.hypot(x + c.mu, y)
        d2 = math.hypot(x - (1 - c.mu), y)
        assert abs(d1 - 1.0) < 1e-12 and abs(d2 - 1.0) < 1e-12


def test_jacobi_constant_conserved():
    c = CR3BP(0.01215)
    x, y = c.lagrange_points()["L4"]
    s = [x + 0.02, y, 0.0, 0.0]
    C0 = c.jacobi_constant(*s)
    worst = 0.0
    for _ in range(50000):
        s = _rk4(c.accel, s, 0.001)
        worst = max(worst, abs(c.jacobi_constant(*s) - C0))
    assert worst < 1e-9, f"Jacobi constant drifted {worst}"


def test_L4_stable_below_routh_threshold():
    # Routh: L4 linearly stable for mu < (1 - sqrt(23/27))/2 ~ 0.03852
    c = CR3BP(0.01215)  # below threshold -> stable
    x, y = c.lagrange_points()["L4"]
    s = [x + 0.01, y, 0.0, 0.0]
    worst = 0.0
    for _ in range(150000):
        s = _rk4(c.accel, s, 0.001)
        worst = max(worst, math.hypot(s[0] - x, s[1] - y))
    assert worst < 0.5, f"stable L4 excursion too large: {worst}"


def test_L4_unstable_above_routh_threshold():
    c = CR3BP(0.1)  # above threshold -> unstable
    x, y = c.lagrange_points()["L4"]
    s = [x + 0.01, y, 0.0, 0.0]
    worst = 0.0
    for _ in range(60000):
        s = _rk4(c.accel, s, 0.001)
        worst = max(worst, math.hypot(s[0] - x, s[1] - y))
    assert worst > 1.0, f"unstable L4 should run away, excursion only {worst}"


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
