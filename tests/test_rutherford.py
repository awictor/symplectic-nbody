"""Rutherford-scattering tests.

Claims checked:
  1. The differential cross section follows 1/sin^4(theta/2) -- huge at small angles,
     small but nonzero at back-scattering (the plum-pudding-killing hard bounces).
  2. A 5 MeV alpha on gold approaches within ~45 fm head-on (Rutherford's nuclear-size
     bound).
  3. Smaller impact parameter -> larger scattering angle; the mapping inverts.
  4. Cross section falls as 1/E^2 with beam energy.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rutherford import (coulomb_constant, differential_cross_section,  # noqa: E402
                        impact_parameter, closest_approach, scattering_angle,
                        MEV, FM)

Z_ALPHA, Z_GOLD = 2, 79
E5 = 5 * MEV


def test_sin4_dependence():
    r = (differential_cross_section(math.radians(90), Z_ALPHA, Z_GOLD, E5)
         / differential_cross_section(math.radians(60), Z_ALPHA, Z_GOLD, E5))
    expected = math.sin(math.radians(30)) ** 4 / math.sin(math.radians(45)) ** 4
    assert abs(r - expected) < 1e-6


def test_backscatter_nonzero():
    assert differential_cross_section(math.radians(179), Z_ALPHA, Z_GOLD, E5) > 0.0


def test_small_angle_diverges():
    small = differential_cross_section(math.radians(1), Z_ALPHA, Z_GOLD, E5)
    big = differential_cross_section(math.radians(90), Z_ALPHA, Z_GOLD, E5)
    assert small > 1e6 * big


def test_closest_approach_gold():
    r = closest_approach(E5, Z_ALPHA, Z_GOLD) / FM
    assert 30.0 < r < 60.0, f"closest approach {r} fm off nuclear scale"


def test_smaller_b_larger_angle():
    a1 = scattering_angle(50 * FM, Z_ALPHA, Z_GOLD, E5)
    a2 = scattering_angle(10 * FM, Z_ALPHA, Z_GOLD, E5)
    assert a2 > a1


def test_impact_angle_invert():
    for deg in (30.0, 90.0, 150.0):
        b = impact_parameter(math.radians(deg), Z_ALPHA, Z_GOLD, E5)
        assert abs(math.degrees(scattering_angle(b, Z_ALPHA, Z_GOLD, E5)) - deg) < 1e-6


def test_cross_section_inverse_E_squared():
    lo = differential_cross_section(math.radians(90), Z_ALPHA, Z_GOLD, E5)
    hi = differential_cross_section(math.radians(90), Z_ALPHA, Z_GOLD, 2 * E5)
    assert abs(lo / hi - 4.0) < 1e-6


def test_closest_approach_inverse_E():
    assert abs(closest_approach(E5, Z_ALPHA, Z_GOLD) / closest_approach(2 * E5, Z_ALPHA, Z_GOLD) - 2.0) < 1e-9


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
