"""Roche-lobe / binary-mass-transfer tests.

Claims checked:
  1. The Eggleton Roche-lobe radius is ~0.38 a for an equal-mass binary and
     grows monotonically with the star's mass ratio.
  2. L1 sits midway (0.5 a) for equal masses and shifts toward the lighter star.
  3. Conservative mass transfer is stable from a less-massive donor (orbit
     widens, d ln a/d ln M_d < 0) and unstable from a more-massive donor.
  4. The orbit-response formula d ln a/d ln M_d = 2(q-1) has the right sign.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roche_lobe import (eggleton_radius, l1_distance,  # noqa: E402
                        conservative_orbit_response, transfer_is_stable)


def test_equal_mass_lobe():
    assert abs(eggleton_radius(1.0) - 0.38) < 0.01, "equal-mass Roche lobe ~0.38 a"


def test_lobe_monotonic_in_q():
    assert eggleton_radius(0.1) < eggleton_radius(1.0) < eggleton_radius(10.0)


def test_l1_midpoint_equal_mass():
    assert abs(l1_distance(1.0) - 0.5) < 1e-9
    # L1 shifts toward the lighter star as q changes
    assert l1_distance(0.5) > 0.5 and l1_distance(2.0) < 0.5


def test_stability_flips_at_equal_mass():
    assert transfer_is_stable(0.5), "less-massive donor -> stable"
    assert not transfer_is_stable(2.0), "more-massive donor -> unstable"


def test_orbit_response_sign():
    assert conservative_orbit_response(0.5) < 0.0, "orbit widens for q<1"
    assert conservative_orbit_response(2.0) > 0.0, "orbit shrinks for q>1"
    assert abs(conservative_orbit_response(1.0)) < 1e-12, "marginal at q=1"


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
