"""Kerr (rotating) black-hole tests.

Claims checked:
  1. At zero spin Kerr reduces to Schwarzschild: horizon 2M, ISCO 6M (both
     prograde and retrograde coincide).
  2. At extremal spin a=M the horizon shrinks to M, the prograde ISCO to 1M, and
     the retrograde ISCO grows to 9M -- the Bardeen-Press-Teukolsky limits.
  3. Spinning up the hole shrinks the prograde ISCO and grows the retrograde one.
  4. The equatorial ergosphere is 2M for any spin, and lies outside the horizon.
  5. a > M is rejected (cosmic censorship: no naked singularity).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kerr import (horizons, ergosphere_radius, isco_radius,  # noqa: E402
                  frame_dragging_omega, horizon_angular_velocity)


def test_zero_spin_is_schwarzschild():
    rp, rm = horizons(0.0)
    assert abs(rp - 2.0) < 1e-12 and abs(rm) < 1e-12
    assert abs(isco_radius(0.0, prograde=True) - 6.0) < 1e-6
    assert abs(isco_radius(0.0, prograde=False) - 6.0) < 1e-6


def test_extremal_limits():
    rp, rm = horizons(1.0)
    assert abs(rp - 1.0) < 1e-9 and abs(rm - 1.0) < 1e-9
    assert abs(isco_radius(1.0, prograde=True) - 1.0) < 1e-6, "extremal prograde ISCO -> 1M"
    assert abs(isco_radius(1.0, prograde=False) - 9.0) < 1e-6, "extremal retrograde ISCO -> 9M"


def test_prograde_shrinks_retrograde_grows():
    for a in (0.3, 0.6, 0.9):
        assert isco_radius(a, True) < 6.0, f"prograde ISCO should shrink below 6M at a={a}"
        assert isco_radius(a, False) > 6.0, f"retrograde ISCO should grow above 6M at a={a}"
    # monotonic in spin
    assert isco_radius(0.9, True) < isco_radius(0.5, True)
    assert isco_radius(0.9, False) > isco_radius(0.5, False)


def test_ergosphere_outside_horizon():
    for a in (0.3, 0.6, 0.9):
        assert abs(ergosphere_radius(a) - 2.0) < 1e-12, "equatorial ergosphere is 2M"
        assert ergosphere_radius(a) >= horizons(a)[0], "ergosphere outside horizon"


def test_frame_dragging_positive_and_decays():
    assert frame_dragging_omega(3.0, 0.9) > frame_dragging_omega(10.0, 0.9) > 0.0
    assert frame_dragging_omega(5.0, 0.0) == 0.0, "no frame dragging without spin"


def test_naked_singularity_rejected():
    try:
        horizons(1.5)
        assert False, "a > M should raise"
    except ValueError:
        pass


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
