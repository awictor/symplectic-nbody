"""Roche-limit / tidal-disruption tests.

Claims checked:
  1. The Roche-limit formula scales correctly: d proportional to R_primary and
     to (rho_primary/rho_sat)^{1/3}.
  2. A rubble-pile satellite well INSIDE the Roche limit is tidally disrupted
     (bound fraction collapses), while one well OUTSIDE stays mostly bound --
     measured at fixed mass/size/time so the difference is purely tidal.
  3. Survival increases monotonically with distance across the Roche limit.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roche import roche_limit, surviving_bound_fraction  # noqa: E402


def test_roche_scaling():
    base = roche_limit(1.0, 1.0, 1.0)
    # linear in R_primary
    assert abs(roche_limit(2.0, 1.0, 1.0) - 2.0 * base) < 1e-12
    # cube-root in density ratio: 8x denser primary -> 2x farther
    assert abs(roche_limit(1.0, 8.0, 1.0) - 2.0 * base) < 1e-12
    # denser satellite -> smaller Roche limit
    assert roche_limit(1.0, 1.0, 8.0) < base


def test_inside_roche_disrupts():
    inside = surviving_bound_fraction(0.4)
    assert inside < 0.2, f"deep inside Roche should disrupt, bound={inside}"


def test_outside_roche_survives():
    outside = surviving_bound_fraction(3.0)
    assert outside > 0.8, f"well outside Roche should survive, bound={outside}"


def test_survival_increases_with_distance():
    fracs = [surviving_bound_fraction(f) for f in (0.4, 1.0, 2.5)]
    assert fracs[0] < fracs[1] < fracs[2], f"survival should grow with distance: {fracs}"


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
