"""Magnetic-mirror / adiabatic-invariant tests.

Claims checked:
  1. The magnetic moment mu = m v_perp^2 / 2B is conserved as B changes (so v_perp
     grows as sqrt(B)).
  2. The loss-cone angle is arcsin(sqrt(1/R_m)); R_m=4 gives exactly 30 degrees.
  3. Large-pitch-angle particles are trapped (mirror); small-pitch-angle ones fall
     into the loss cone and escape.
  4. The mirror point is where B = B_min / sin^2(alpha).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from magnetic_mirror import (magnetic_moment, mirror_ratio,  # noqa: E402
                             loss_cone_angle, is_trapped, mirror_field,
                             perp_velocity_at)


def test_moment_conserved():
    # v_perp ~ sqrt(B) keeps mu fixed
    mu1 = magnetic_moment(1.0, 0.5, 1.0)
    mu2 = magnetic_moment(1.0, 0.5 * math.sqrt(4.0), 4.0)
    assert abs(mu1 - mu2) < 1e-12


def test_loss_cone_30_deg_at_ratio_4():
    a = math.degrees(loss_cone_angle(4.0, 1.0))
    assert abs(a - 30.0) < 1e-9


def test_loss_cone_shrinks_with_ratio():
    assert loss_cone_angle(100.0, 1.0) < loss_cone_angle(10.0, 1.0) < loss_cone_angle(2.0, 1.0)


def test_large_pitch_trapped():
    assert is_trapped(math.radians(80.0), 10.0, 1.0)


def test_small_pitch_escapes():
    assert not is_trapped(math.radians(10.0), 10.0, 1.0)


def test_boundary_at_loss_cone():
    R = 10.0
    a_lc = loss_cone_angle(R, 1.0)
    # just inside the loss cone escapes, just outside is trapped
    assert not is_trapped(a_lc * 0.99, R, 1.0)
    assert is_trapped(a_lc * 1.01, R, 1.0)


def test_mirror_field():
    # 30 deg pitch mirrors where B = B_min / sin^2(30) = 4 B_min
    assert abs(mirror_field(1.0, math.radians(30.0)) - 4.0) < 1e-9


def test_perp_velocity_grows_with_field():
    v1 = perp_velocity_at(1.0, math.radians(30.0), 1.0, 1.0)
    v4 = perp_velocity_at(1.0, math.radians(30.0), 1.0, 4.0)
    assert abs(v4 / v1 - 2.0) < 1e-9   # sqrt(4) = 2


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
