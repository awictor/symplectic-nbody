"""Barnes-Hut correctness tests.

The tree code is only useful if it agrees with the exact O(N^2) force. These
tests pin that agreement:

  1. theta = 0 (never approximate) must match direct summation to machine eps.
  2. theta = 0.5 must match direct summation to a few percent (the known B-H
     accuracy regime), and tightening theta must reduce the error.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import plummer_sphere  # noqa: E402
from barnes_hut import BarnesHut  # noqa: E402


def _max_rel_force_error(theta, n=200, seed=7):
    sysn = plummer_sphere(n=n, seed=seed)
    exact = sysn.accel(sysn.pos)
    bh = BarnesHut(G=sysn.G, theta=theta, softening=(sysn.soft2 ** 0.5))
    approx = bh.accel(sysn.pos, sysn.m)
    worst = 0.0
    for i in range(n):
        de = sum((approx[i][k] - exact[i][k]) ** 2 for k in range(3)) ** 0.5
        mag = sum(exact[i][k] ** 2 for k in range(3)) ** 0.5 or 1e-30
        worst = max(worst, de / mag)
    return worst


def test_theta_zero_matches_direct():
    err = _max_rel_force_error(theta=0.0, n=150)
    assert err < 1e-9, f"theta=0 should be exact, got {err}"


def test_theta_half_within_few_percent():
    err = _max_rel_force_error(theta=0.5, n=300)
    assert err < 0.05, f"theta=0.5 error too large: {err}"


def test_tighter_theta_more_accurate():
    loose = _max_rel_force_error(theta=0.8, n=300)
    tight = _max_rel_force_error(theta=0.2, n=300)
    assert tight < loose, f"theta=0.2 ({tight}) should beat theta=0.8 ({loose})"


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
