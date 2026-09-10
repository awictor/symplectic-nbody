"""Parker transonic solar-wind tests.

Claims checked:
  1. The isothermal sound speed of a ~1.5 MK corona is ~140 km/s, and the sonic
     critical radius is a few solar radii.
  2. The transonic solution passes exactly through Mach 1 at the critical radius,
     is subsonic inside it, and supersonic outside.
  3. The wind speed rises monotonically with radius and reaches a few hundred km/s
     by 1 AU.
  4. A hotter corona drives a faster wind.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parker_wind import (sound_speed, critical_radius, wind_mach,  # noqa: E402
                         wind_speed, R_SUN, AU, M_SUN)

T = 1.5e6


def test_sound_speed():
    cs = sound_speed(T) / 1e3
    assert abs(cs - 144.0) < 10.0, f"coronal sound speed {cs} km/s not ~140"


def test_critical_radius_few_solar_radii():
    rc = critical_radius(T) / R_SUN
    assert 2.0 < rc < 8.0, f"critical radius {rc} R_sun off scale"


def test_mach_one_at_critical_radius():
    rc = critical_radius(T)
    assert abs(wind_mach(rc, T) - 1.0) < 1e-3


def test_subsonic_inside_supersonic_outside():
    rc = critical_radius(T)
    assert wind_mach(0.5 * rc, T) < 1.0
    assert wind_mach(3.0 * rc, T) > 1.0


def test_wind_accelerates_outward():
    rc = critical_radius(T)
    v1 = wind_speed(0.7 * rc, T)
    v2 = wind_speed(2.0 * rc, T)
    v3 = wind_speed(AU, T)
    assert v1 < v2 < v3, "wind should accelerate monotonically outward"


def test_speed_at_1au():
    v = wind_speed(AU, T) / 1e3
    assert 300.0 < v < 800.0, f"1 AU wind speed {v} km/s off observed range"


def test_hotter_faster_wind():
    cool = wind_speed(AU, 1.0e6)
    hot = wind_speed(AU, 2.0e6)
    assert hot > cool, "a hotter corona should drive a faster wind"


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
