"""Conservation-law tests. These are the *scientific claims* of the project,
checked automatically:

  1. Symplectic integrators (verlet, forest_ruth) keep total energy bounded.
  2. RK4 drifts secularly and, over a long run, drifts more than verlet.
  3. Linear & angular momentum are conserved to machine precision.
  4. forest_ruth (4th order) beats verlet (2nd order) on max energy error.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import two_body_circular, two_body_eccentric, figure_eight  # noqa: E402


def _energy_stats(system_fn, method, dt, steps):
    sys_ = system_fn()
    e0 = sys_.total_energy()
    max_rel = 0.0
    for _t, e, _L in sys_.run(method, dt, steps, sample_every=10):
        max_rel = max(max_rel, abs((e - e0) / e0))
    return max_rel


def _drift_slope(system_fn, method, dt, steps):
    """Secular drift rate: |slope| of the least-squares line through the relative
    energy-error time series. Symplectic methods oscillate with ~zero slope;
    non-symplectic RK4 has a nonzero slope (energy walks off monotonically).
    Slope is robust to the bounded oscillation that fools a max/final metric."""
    sys_ = system_fn()
    e0 = sys_.total_energy()
    ts, es = [], []
    for t, e, _L in sys_.run(method, dt, steps, sample_every=50):
        ts.append(t)
        es.append((e - e0) / e0)
    n = len(ts)
    mt, me = sum(ts) / n, sum(es) / n
    num = sum((ts[i] - mt) * (es[i] - me) for i in range(n))
    den = sum((ts[i] - mt) ** 2 for i in range(n))
    return abs(num / den)


def test_verlet_energy_bounded():
    max_rel = _energy_stats(two_body_circular, "verlet", 0.01, 20000)
    assert max_rel < 1e-3, f"verlet energy drift too large: {max_rel}"


def test_forest_ruth_beats_verlet():
    v = _energy_stats(two_body_circular, "verlet", 0.01, 20000)
    fr = _energy_stats(two_body_circular, "forest_ruth", 0.01, 20000)
    assert fr < v, f"forest_ruth ({fr}) should beat verlet ({v})"


def test_rk4_drifts_more_than_verlet_long_run():
    # On an eccentric orbit RK4's energy walks off monotonically (nonzero slope),
    # while the symplectic method only oscillates (near-zero slope).
    v = _drift_slope(two_body_eccentric, "verlet", 0.01, 40000)
    r = _drift_slope(two_body_eccentric, "rk4", 0.01, 40000)
    assert r > 3 * v, f"rk4 drift slope ({r}) should exceed verlet ({v})"


def test_momentum_conserved():
    sys_ = figure_eight()
    p0 = sys_.linear_momentum()
    for _ in sys_.run("verlet", 0.001, 5000, sample_every=5000):
        pass
    p1 = sys_.linear_momentum()
    for k in range(3):
        assert abs(p1[k] - p0[k]) < 1e-9, f"momentum component {k} not conserved"


def test_angular_momentum_conserved():
    sys_ = two_body_circular()
    L0 = sys_.angular_momentum()
    for _ in sys_.run("forest_ruth", 0.01, 10000, sample_every=10000):
        pass
    L1 = sys_.angular_momentum()
    assert abs(L1[2] - L0[2]) < 1e-8, "z angular momentum not conserved"


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
