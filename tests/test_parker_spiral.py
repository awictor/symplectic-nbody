"""Parker-spiral tests.

Claims checked:
  1. The garden-hose angle at 1 AU (400 km/s wind) is ~45 degrees.
  2. The field is nearly radial close to the Sun and nearly azimuthal far out.
  3. Flux conservation: B_r ~ 1/r^2, B_phi ~ 1/r, so |B| -> azimuthal far away.
  4. A faster wind winds the spiral less tightly (smaller angle).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parker_spiral import (spiral_angle, phi_of_r, field_components,  # noqa: E402
                           field_magnitude, AU, R_SUN, OMEGA_SUN)

U = 4e5  # 400 km/s typical solar wind


def test_angle_at_1au():
    psi = math.degrees(spiral_angle(AU, U))
    assert abs(psi - 45.0) < 5.0, f"1 AU spiral angle {psi} deg not ~45"


def test_nearly_radial_near_sun():
    psi = math.degrees(spiral_angle(0.1 * AU, U))
    assert psi < 10.0, f"field should be nearly radial near Sun, got {psi} deg"


def test_nearly_azimuthal_far_out():
    psi = math.degrees(spiral_angle(5.2 * AU, U))  # Jupiter
    assert psi > 75.0, f"field should be nearly azimuthal at Jupiter, got {psi} deg"


def test_radial_inverse_square():
    B_r1, _ = field_components(AU, U, 5e-9)
    B_r2, _ = field_components(2 * AU, U, 5e-9)
    assert abs(B_r1 / B_r2 - 4.0) < 1e-9, "B_r should scale as 1/r^2"


def test_azimuthal_inverse_r():
    _, Bp1 = field_components(AU, U, 5e-9)
    _, Bp2 = field_components(2 * AU, U, 5e-9)
    assert abs(Bp1 / Bp2 - 2.0) < 1e-9, "B_phi should scale as 1/r"


def test_faster_wind_less_wound():
    slow = spiral_angle(AU, 3e5)
    fast = spiral_angle(AU, 8e5)
    assert fast < slow, "a faster wind should wind the spiral less"


def test_phi_decreases_outward():
    # azimuth trails behind (decreases) with distance for prograde rotation
    inner = phi_of_r(AU, U)
    outer = phi_of_r(5 * AU, U)
    assert outer < inner


def test_magnitude_is_hypot():
    B_r, B_phi = field_components(3 * AU, U, 5e-9)
    assert abs(field_magnitude(3 * AU, U, 5e-9) - math.hypot(B_r, B_phi)) < 1e-30


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
