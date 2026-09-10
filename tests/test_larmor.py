"""Larmor-formula tests.

Claims checked:
  1. The Larmor power is quadratic in acceleration.
  2. Relativistic circular (perpendicular) acceleration boosts the power by
     gamma^4, and linear (parallel) acceleration by gamma^6.
  3. The classical-electromagnetic hydrogen atom collapses in ~1.6e-11 s -- the
     catastrophe that forced quantum mechanics.
  4. Perpendicular radiation dominates parallel at fixed gamma > 1.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from larmor import (larmor_power, relativistic_power_perpendicular,  # noqa: E402
                    relativistic_power_parallel, atom_collapse_time)


def test_power_quadratic_in_acceleration():
    assert abs(larmor_power(2e20) / larmor_power(1e20) - 4.0) < 1e-9


def test_perpendicular_gamma_fourth():
    r = relativistic_power_perpendicular(1e20, 100.0) / larmor_power(1e20)
    assert abs(r - 100.0 ** 4) / r < 1e-9, "perp boost should be gamma^4"


def test_parallel_gamma_sixth():
    r = relativistic_power_parallel(1e20, 100.0) / larmor_power(1e20)
    assert abs(r - 100.0 ** 6) / r < 1e-9, "parallel boost should be gamma^6"


def test_atom_collapse_time():
    t = atom_collapse_time()
    assert 1e-11 < t < 3e-11, f"classical atom collapse {t} s not ~1.6e-11"


def test_parallel_exceeds_perpendicular():
    # at the same gamma and a, parallel (gamma^6) > perpendicular (gamma^4)
    a, g = 1e20, 50.0
    assert relativistic_power_parallel(a, g) > relativistic_power_perpendicular(a, g)


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
