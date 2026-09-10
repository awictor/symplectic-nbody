"""Hohmann-transfer tests against standard mission-design numbers.

Claims checked:
  1. LEO (200 km) -> GEO (35786 km) needs ~3.9 km/s of total delta-v over ~5.3
     hours.
  2. Earth -> Mars (heliocentric) needs ~5.6 km/s and takes ~259 days, with a
     ~44 deg launch phase angle -- the actual Mars-mission window.
  3. Both burns are prograde (positive dv) when raising the orbit, and the
     transfer time is half the transfer-ellipse period.
  4. Tsiolkovsky's rocket equation gives a sensible mass ratio for a real
     delta-v and exhaust velocity.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hohmann import (transfer, phase_angle, rocket_equation_mass_ratio,  # noqa: E402
                     MU_EARTH, MU_SUN, AU, DAY)

RE = 6.378e6


def test_leo_to_geo():
    r1, r2 = RE + 200e3, RE + 35786e3
    dv1, dv2, dvt, t = transfer(r1, r2, MU_EARTH)
    assert abs(dvt - 3900.0) < 150.0, f"LEO->GEO total dv {dvt} not ~3.9 km/s"
    assert abs(t / 3600.0 - 5.3) < 0.2, f"LEO->GEO time {t/3600} not ~5.3 hr"
    assert dv1 > 0 and dv2 > 0, "raising an orbit needs two prograde burns"


def test_earth_to_mars():
    dv1, dv2, dvt, t = transfer(1.0 * AU, 1.524 * AU, MU_SUN)
    assert abs(dvt - 5600.0) < 200.0, f"Earth->Mars total dv {dvt} not ~5.6 km/s"
    assert abs(t / DAY - 259.0) < 5.0, f"Earth->Mars time {t/DAY} not ~259 days"


def test_mars_phase_angle():
    alpha = math.degrees(phase_angle(1.0 * AU, 1.524 * AU, MU_SUN))
    assert abs(alpha - 44.0) < 2.0, f"Mars launch phase angle {alpha} not ~44 deg"


def test_transfer_time_is_half_ellipse_period():
    r1, r2 = RE + 200e3, RE + 20000e3
    _dv1, _dv2, _dvt, t = transfer(r1, r2, MU_EARTH)
    a_t = 0.5 * (r1 + r2)
    full_period = 2.0 * math.pi * math.sqrt(a_t ** 3 / MU_EARTH)
    assert abs(t - 0.5 * full_period) < 1e-3, "transfer time should be half the ellipse period"


def test_rocket_equation():
    # a 3.9 km/s transfer with a 4.4 km/s (LH2/LOX) exhaust needs m0/mf ~ 2.4
    ratio = rocket_equation_mass_ratio(3900.0, 4400.0)
    assert 2.0 < ratio < 3.0, f"mass ratio {ratio} off"
    # larger dv -> larger required ratio
    assert rocket_equation_mass_ratio(6000.0, 4400.0) > ratio


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
