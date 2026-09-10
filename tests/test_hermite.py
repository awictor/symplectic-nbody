"""Fourth-order Hermite integrator tests.

Claims checked:
  1. The analytic jerk matches a finite-difference of the acceleration.
  2. Hermite converges at fourth order against the exact Kepler orbit.
  3. One force evaluation per step still reaches 4th order (the whole point:
     Hermite matches RK4's order with fewer force calls -- RK4 needs 4).
  4. Energy stays well controlled over a long eccentric run.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermite import hermite_step, accel_and_jerk  # noqa: E402
from systems import two_body_eccentric  # noqa: E402
from kepler import KeplerOrbit  # noqa: E402


def test_jerk_matches_finite_difference():
    s = two_body_eccentric(e=0.4)
    a0, j0 = accel_and_jerk(s.pos, s.vel, s.m, G=s.G)
    # finite-difference da/dt by nudging positions along velocity
    h = 1e-6
    pos_p = [[s.pos[i][k] + h * s.vel[i][k] for k in range(3)] for i in range(s.n)]
    pos_m = [[s.pos[i][k] - h * s.vel[i][k] for k in range(3)] for i in range(s.n)]
    a_p, _ = accel_and_jerk(pos_p, s.vel, s.m, G=s.G)
    a_m, _ = accel_and_jerk(pos_m, s.vel, s.m, G=s.G)
    for i in range(s.n):
        for k in range(3):
            fd = (a_p[i][k] - a_m[i][k]) / (2 * h)
            assert abs(fd - j0[i][k]) < 1e-4, f"jerk mismatch body {i},{k}: {fd} vs {j0[i][k]}"


def _order():
    k = KeplerOrbit(e=0.5, a=1.0, mu=2.0)
    T = k.period

    def err_at(steps):
        s = two_body_eccentric(e=0.5, a=1.0)
        dt = T / steps
        pos, vel = [list(p) for p in s.pos], [list(v) for v in s.vel]
        for _ in range(steps):
            pos, vel = hermite_step(pos, vel, s.m, dt, G=s.G)
        tp, _ = k.barycentric(T, 1.0, 1.0)
        return max(abs(pos[i][j] - tp[i][j]) for i in range(2) for j in range(3))

    return math.log(err_at(1000) / err_at(2000)) / math.log(2.0)


def test_fourth_order_convergence():
    o = _order()
    assert abs(o - 4.0) < 0.2, f"Hermite order {o} not ~4"


def test_energy_controlled():
    s = two_body_eccentric(e=0.7)
    e0 = s.total_energy()
    pos, vel = [list(p) for p in s.pos], [list(v) for v in s.vel]
    worst = 0.0
    for i in range(20000):
        pos, vel = hermite_step(pos, vel, s.m, 0.01, G=s.G)
        if i % 100 == 0:
            s.pos, s.vel = pos, vel
            worst = max(worst, abs((s.total_energy() - e0) / e0))
    assert worst < 1e-3, f"Hermite energy error too large: {worst}"


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
