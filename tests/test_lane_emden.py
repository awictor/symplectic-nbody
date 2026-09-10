"""Lane-Emden stellar-structure tests.

Claims checked:
  1. n=0: theta = 1 - xi^2/6 with surface xi_1 = sqrt(6).
  2. n=1: theta = sin(xi)/xi with surface xi_1 = pi (profile to ~1e-9).
  3. n=5: theta = 1/sqrt(1 + xi^2/3), which never reaches zero (infinite radius).
  4. n=3 (the Eddington standard model): xi_1 ~ 6.897 and the mass integral
     -xi_1^2 theta'(xi_1) ~ 2.018 -- the standard tabulated values.
  5. theta starts at 1 with zero slope (the regular center).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lane_emden import (solve, analytic_n0, analytic_n1, analytic_n5)  # noqa: E402


def test_n0_surface_and_profile():
    xis, th, xi1, mass = solve(0.0)
    assert abs(xi1 - math.sqrt(6.0)) < 1e-3, f"n=0 xi_1 {xi1} not sqrt(6)"
    err = max(abs(th[i] - analytic_n0(xis[i])) for i in range(0, len(xis), 20))
    assert err < 1e-6, f"n=0 profile error {err}"


def test_n1_surface_is_pi():
    xis, th, xi1, mass = solve(1.0)
    assert abs(xi1 - math.pi) < 1e-3, f"n=1 xi_1 {xi1} not pi"
    err = max(abs(th[i] - analytic_n1(xis[i])) for i in range(0, len(xis), 20))
    assert err < 1e-6, f"n=1 profile error {err}"


def test_n5_has_infinite_radius():
    xis, th, xi1, mass = solve(5.0, xi_max=15.0)
    assert xi1 == float("inf"), f"n=5 should not reach a surface, got xi_1={xi1}"
    err = max(abs(th[i] - analytic_n5(xis[i])) for i in range(0, len(xis), 20))
    assert err < 1e-6, f"n=5 profile error {err}"


def test_n3_standard_model():
    xis, th, xi1, mass = solve(3.0)
    assert abs(xi1 - 6.897) < 0.01, f"n=3 xi_1 {xi1} not ~6.897"
    assert abs(mass - 2.018) < 0.01, f"n=3 mass factor {mass} not ~2.018"


def test_regular_center():
    xis, th, xi1, mass = solve(1.5)
    assert abs(th[0] - 1.0) < 1e-6, "theta(0) should be 1"
    # near-center profile flat (zero slope): theta barely below 1 at small xi
    assert th[1] > 0.999, "profile should start with near-zero slope"


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
