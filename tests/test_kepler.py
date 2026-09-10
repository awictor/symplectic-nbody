"""Kepler analytic-solution tests: the ground-truth yardstick.

Claims checked:
  1. Kepler's equation solver inverts M = E - e sin E to machine precision.
  2. The analytic initial state matches systems.two_body_eccentric exactly.
  3. Every integrator converges to the exact Kepler orbit at its theoretical
     order: verlet ~2, forest_ruth ~4, rk4 ~4 (measured, not assumed).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kepler import KeplerOrbit, solve_kepler  # noqa: E402
from systems import two_body_eccentric  # noqa: E402
from integrators import INTEGRATORS  # noqa: E402


def test_kepler_equation_inverts():
    for e in (0.0, 0.3, 0.7, 0.95):
        for M in (0.1, 1.0, 3.0, 5.0):
            E = solve_kepler(M, e)
            residual = abs((E - e * math.sin(E)) - math.fmod(M, 2 * math.pi))
            assert residual < 1e-12, f"kepler residual {residual} at e={e}, M={M}"


def test_analytic_matches_initial_conditions():
    k = KeplerOrbit(e=0.7, a=1.0, mu=2.0)
    pos, vel = k.barycentric(0.0, 1.0, 1.0)
    s = two_body_eccentric(e=0.7, a=1.0)
    for i in range(2):
        for j in range(3):
            assert abs(pos[i][j] - s.pos[i][j]) < 1e-12
            assert abs(vel[i][j] - s.vel[i][j]) < 1e-12


def _order(method, e=0.5):
    k = KeplerOrbit(e=e, a=1.0, mu=2.0)
    T = k.period

    def err_at(steps):
        s = two_body_eccentric(e=e, a=1.0)
        dt = T / steps
        pos, vel = [list(p) for p in s.pos], [list(v) for v in s.vel]
        integ = INTEGRATORS[method]
        for _ in range(steps):
            pos, vel = integ(pos, vel, s.accel, dt)
        tp, _ = k.barycentric(T, 1.0, 1.0)
        return max(abs(pos[i][j] - tp[i][j]) for i in range(2) for j in range(3))

    e1, e2 = err_at(1000), err_at(2000)
    return math.log(e1 / e2) / math.log(2.0)


def test_verlet_is_second_order():
    o = _order("verlet")
    assert abs(o - 2.0) < 0.2, f"verlet order {o} not ~2"


def test_forest_ruth_is_fourth_order():
    o = _order("forest_ruth")
    assert abs(o - 4.0) < 0.3, f"forest_ruth order {o} not ~4"


def test_rk4_is_fourth_order():
    o = _order("rk4")
    assert abs(o - 4.0) < 0.3, f"rk4 order {o} not ~4"


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
