"""Escape / cosmic-velocity tests.

Claims checked:
  1. Earth's first cosmic speed (LEO) is ~7.9 km/s and its escape speed ~11.2
     km/s.
  2. Escape speed is always sqrt(2) times the circular-orbit speed.
  3. The heliocentric escape speed at Earth's orbit (~42 km/s) is the third
     cosmic speed for leaving the Solar System.
  4. Setting the escape speed to c recovers the Schwarzschild radius: ~2954 m for
     the Sun, ~8.9 mm for the Earth.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cosmic_velocities import (orbital_speed, escape_speed,  # noqa: E402
                               escape_over_orbital,
                               solar_system_escape_from_earth_orbit,
                               schwarzschild_radius_from_escape,
                               M_EARTH, R_EARTH, M_SUN, C)


def test_earth_orbital_speed():
    assert abs(orbital_speed(M_EARTH, R_EARTH) / 1e3 - 7.9) < 0.1


def test_earth_escape_speed():
    assert abs(escape_speed(M_EARTH, R_EARTH) / 1e3 - 11.2) < 0.1


def test_escape_orbital_ratio():
    assert abs(escape_over_orbital() - math.sqrt(2.0)) < 1e-12
    assert abs(escape_speed(M_EARTH, R_EARTH) / orbital_speed(M_EARTH, R_EARTH)
               - math.sqrt(2.0)) < 1e-12


def test_solar_system_escape():
    v = solar_system_escape_from_earth_orbit() / 1e3
    assert abs(v - 42.1) < 0.5, f"solar-system escape {v} km/s not ~42"


def test_schwarzschild_radius():
    rs_sun = schwarzschild_radius_from_escape(M_SUN)
    assert abs(rs_sun - 2954.0) < 5.0, f"Sun R_s {rs_sun} m not ~2954"
    rs_earth = schwarzschild_radius_from_escape(M_EARTH) * 1000
    assert abs(rs_earth - 8.9) < 0.3, f"Earth R_s {rs_earth} mm not ~8.9"
    # escape speed at R_s equals c
    assert abs(escape_speed(M_SUN, rs_sun) - C) < 1.0


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
