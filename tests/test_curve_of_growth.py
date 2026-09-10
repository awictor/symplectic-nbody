"""Curve-of-growth tests.

Claims checked:
  1. Weak lines are linear: W ~ N (equivalently ~ tau0).
  2. Saturated lines are flat: W barely grows over orders of magnitude in tau0.
  3. Damped lines recover square-root growth: W ~ sqrt(N), so 100x column -> 10x W.
  4. The equivalent width rises monotonically with column density throughout, and
     the regime labels are ordered linear -> saturated -> damped.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from curve_of_growth import (central_optical_depth, equivalent_width,  # noqa: E402
                             regime, column_from_linear_width)


def test_linear_regime():
    assert regime(0.01) == "linear"
    # doubling tau0 doubles W in the linear regime
    assert abs(equivalent_width(0.02) / equivalent_width(0.01) - 2.0) < 1e-9


def test_saturated_is_flat():
    # 1000x in tau0 gives only a modest rise on the flat part
    w1 = equivalent_width(10.0)
    w2 = equivalent_width(1e4)
    assert w2 / w1 < 3.0, "saturated part should be nearly flat"
    assert regime(100.0) == "saturated"


def test_damped_sqrt():
    # 100x column -> ~10x equivalent width in the damping regime
    assert abs(equivalent_width(1e8) / equivalent_width(1e6) - 10.0) < 0.1
    assert regime(1e7) == "damped"


def test_monotonic():
    taus = [10 ** (-2 + 0.5 * i) for i in range(0, 22)]
    Ws = [equivalent_width(t) for t in taus]
    assert all(Ws[i + 1] >= Ws[i] - 1e-9 for i in range(len(Ws) - 1))


def test_regime_ordering():
    assert regime(0.1) == "linear"
    assert regime(50.0) == "saturated"
    assert regime(1e8) == "damped"


def test_optical_depth_proportional_to_column():
    t1 = central_optical_depth(100.0, 0.5, 2.0)
    t2 = central_optical_depth(200.0, 0.5, 2.0)
    assert abs(t2 / t1 - 2.0) < 1e-9


def test_column_inversion():
    N = 0.3
    W = equivalent_width(central_optical_depth(N, 0.5, 2.0))
    N_rec = column_from_linear_width(W, 0.5, 2.0)
    assert abs(N_rec - N) < 1e-6


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
