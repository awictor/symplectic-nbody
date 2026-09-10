"""Precession-of-the-equinoxes tests.

Claims checked:
  1. The combined luni-solar precession is ~50.3 arcsec/yr.
  2. That implies a ~25,800-year period for a full circuit of the pole.
  3. The Moon's torque beats the Sun's by a factor ~2.2 (nearby M/r^3 wins).
  4. The rate scales as M/r^3 and as cos(obliquity), and vanishes at 90 deg tilt.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from axial_precession import (torque_precession_rate, lunisolar_rate,  # noqa: E402
                        precession_period_years, rate_arcsec_per_year,
                        moon_to_sun_ratio, M_SUN, M_MOON, AU, R_MOON, OBLIQUITY)


def test_lunisolar_arcsec():
    a = rate_arcsec_per_year(lunisolar_rate())
    assert abs(a - 50.3) < 1.5, f"luni-solar {a} arcsec/yr not ~50.3"


def test_period():
    yr = precession_period_years(lunisolar_rate())
    assert abs(yr - 25800.0) < 1000.0, f"period {yr} yr not ~25800"


def test_moon_beats_sun():
    r = moon_to_sun_ratio()
    assert abs(r - 2.2) < 0.3, f"moon/sun torque ratio {r} not ~2.2"
    # and the direct rates agree with the ratio
    sun = torque_precession_rate(M_SUN, AU)
    moon = torque_precession_rate(M_MOON, R_MOON)
    assert abs(moon / sun - r) < 1e-9


def test_inverse_cube_scaling():
    # halving the distance raises the torque rate 8x
    near = torque_precession_rate(M_SUN, AU / 2.0)
    far = torque_precession_rate(M_SUN, AU)
    assert abs(near / far - 8.0) < 1e-9


def test_obliquity_dependence():
    # rate ~ cos(eps): zero at 90 deg, larger for a smaller tilt
    upright = torque_precession_rate(M_SUN, AU, eps=math.radians(10.0))
    tilted = torque_precession_rate(M_SUN, AU, eps=OBLIQUITY)
    perp = torque_precession_rate(M_SUN, AU, eps=math.pi / 2.0)
    assert upright > tilted
    assert abs(perp) < 1e-6 * tilted, "no precession when spin axis lies in the orbit plane"


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
