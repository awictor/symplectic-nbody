"""J2-orbital-precession tests.

Claims checked:
  1. A low Earth orbit at the ISS inclination (51.6 deg) regresses its nodes by
     several degrees per day (westward, negative).
  2. The sun-synchronous inclination (~98 deg) makes the nodal drift exactly match
     Earth's ~0.9856 deg/day motion around the Sun.
  3. The critical inclination where apsidal precession vanishes is 63.43 deg.
  4. Nodal regression is zero for a polar orbit (cos 90 = 0) and maximal equatorial.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j2_precession import (mean_motion, nodal_precession_rate,  # noqa: E402
                           apsidal_precession_rate, sun_synchronous_inclination,
                           critical_inclination, DEG_PER_DAY, SUN_RATE_DEG_DAY,
                           R_EARTH)

A_LEO = 7000e3


def test_iss_nodal_drift():
    rate = nodal_precession_rate(A_LEO, 0.0, math.radians(51.6)) * DEG_PER_DAY
    assert -6.0 < rate < -3.0, f"ISS nodal drift {rate} deg/day off scale"


def test_prograde_regresses_westward():
    assert nodal_precession_rate(A_LEO, 0.0, math.radians(30.0)) < 0.0


def test_sun_synchronous():
    a = (6378 + 700) * 1e3
    i = sun_synchronous_inclination(a)
    assert 96.0 < math.degrees(i) < 101.0
    # its nodal drift matches the Sun's apparent motion
    rate = nodal_precession_rate(a, 0.0, i) * DEG_PER_DAY
    assert abs(rate - SUN_RATE_DEG_DAY) < 1e-3


def test_sun_sync_is_retrograde():
    i = sun_synchronous_inclination((6378 + 700) * 1e3)
    assert math.degrees(i) > 90.0   # requires a retrograde inclination


def test_critical_inclination():
    assert abs(math.degrees(critical_inclination()) - 63.43) < 0.05


def test_apsidal_zero_at_critical():
    ic = critical_inclination()
    assert abs(apsidal_precession_rate(A_LEO, 0.0, ic)) < 1e-15


def test_polar_no_nodal_drift():
    assert abs(nodal_precession_rate(A_LEO, 0.0, math.pi / 2.0)) < 1e-20


def test_apsidal_positive_equatorial():
    # low-inclination orbits advance the perigee (5cos^2 i - 1 > 0)
    assert apsidal_precession_rate(A_LEO, 0.0, 0.0) > 0.0


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
