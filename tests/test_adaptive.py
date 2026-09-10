"""Adaptive Dormand-Prince RK45 tests.

Claims checked:
  1. Hits a known analytic target: after one full period the eccentric two-body
     system returns to its start position (closed orbit) to a tight tolerance.
  2. Step-size control actually adapts: the minimum accepted step (pericenter)
     is much smaller than the maximum (apocenter).
  3. Efficiency: to reach a fixed trajectory accuracy on an eccentric orbit,
     adaptive DP45 uses far fewer force evaluations than fixed-step RK4.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import two_body_eccentric  # noqa: E402
from adaptive import DormandPrince  # noqa: E402


def _period(a=1.0, mu=2.0):
    # Kepler's third law with G=1: T = 2*pi*sqrt(a^3/mu)
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def test_closed_orbit_returns_to_start():
    s = two_body_eccentric(e=0.7, a=1.0)
    start = [list(p) for p in s.pos]
    dp = DormandPrince(s.accel, s.n, rtol=1e-10, atol=1e-13, h_init=1e-3)
    pos, vel = dp.integrate(s.pos, s.vel, _period())
    err = max(abs(pos[i][k] - start[i][k]) for i in range(s.n) for k in range(3))
    assert err < 1e-5, f"orbit did not close: max pos error {err}"


def test_stepsize_adapts():
    s = two_body_eccentric(e=0.8, a=1.0)
    steps = []
    last_t = [0.0]

    def sample(t, p, v):
        steps.append(t - last_t[0])
        last_t[0] = t

    dp = DormandPrince(s.accel, s.n, rtol=1e-9, h_init=1e-3)
    dp.integrate(s.pos, s.vel, _period(), on_sample=sample)
    hmin, hmax = min(steps), max(steps)
    assert hmax / hmin > 5.0, f"step size barely adapted: hmax/hmin={hmax/hmin}"


def test_adaptive_beats_fixed_rk4_efficiency():
    # Reference: extremely tight adaptive run gives the "true" end state.
    s0 = two_body_eccentric(e=0.7, a=1.0)
    ref_dp = DormandPrince(s0.accel, s0.n, rtol=1e-13, atol=1e-15, h_init=1e-4)
    ref_pos, _ = ref_dp.integrate(s0.pos, s0.vel, _period())

    def end_error(pos):
        return max(abs(pos[i][k] - ref_pos[i][k]) for i in range(len(pos)) for k in range(3))

    # Adaptive at a modest tolerance.
    s1 = two_body_eccentric(e=0.7, a=1.0)
    dp = DormandPrince(s1.accel, s1.n, rtol=1e-8, atol=1e-11, h_init=1e-3)
    p_ad, _ = dp.integrate(s1.pos, s1.vel, _period())
    err_ad = end_error(p_ad)
    feval_ad = dp.n_feval

    # Fixed RK4 tuned to match that accuracy; count its force evaluations.
    from integrators import rk4  # noqa: E402
    target = _period()
    best_feval = None
    for steps in (2000, 4000, 8000, 16000, 32000, 64000):
        s2 = two_body_eccentric(e=0.7, a=1.0)
        dt = target / steps
        pos, vel = [list(p) for p in s2.pos], [list(v) for v in s2.vel]
        for _ in range(steps):
            pos, vel = rk4(pos, vel, s2.accel, dt)
        if end_error(pos) <= err_ad:
            best_feval = steps * 4  # RK4 = 4 force evals/step
            break

    assert best_feval is not None, "fixed RK4 never matched adaptive accuracy in range"
    assert feval_ad < best_feval, (
        f"adaptive feval {feval_ad} should beat fixed RK4 feval {best_feval} "
        f"at accuracy {err_ad:.2e}")


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
