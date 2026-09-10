"""Tidal-locking tests.

Claims checked:
  1. The Moon locks to Earth well within the solar system's age; the Earth does
     not lock to the far weaker lunar tide in the same time.
  2. The locking time scales as a^6 (distance) and as 1/M_p^2 (primary mass).
  3. The max-lock distance sits beyond the Moon's orbit (so the Moon is inside
     the locking zone) and shrinks for a weaker primary.
  4. is_locked agrees with comparing t_lock to the budget.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tidal_locking import (locking_time, moment_of_inertia, is_locked,  # noqa: E402
                           max_locking_distance, M_EARTH, M_MOON, R_MOON,
                           R_EARTH, A_MOON, AGE_SOLAR_SYSTEM, GYR)

WI_MOON = 2 * math.pi / (5 * 3600.0)   # fast primordial spin, 5-hour period
I_MOON = moment_of_inertia(M_MOON, R_MOON)


def test_moon_locks_fast():
    t = locking_time(WI_MOON, A_MOON, R_MOON, I_MOON, M_EARTH) / GYR
    assert t < 1.0, f"Moon should lock within a Gyr, got {t} Gyr"
    assert is_locked(WI_MOON, A_MOON, R_MOON, I_MOON, M_EARTH)


def test_earth_not_locked():
    wi = 2 * math.pi / (24 * 3600.0)
    I = moment_of_inertia(M_EARTH, R_EARTH, 0.33)
    # Earth despinning against the Moon's weaker tide
    assert not is_locked(wi, A_MOON, R_EARTH, I, M_MOON)


def test_a6_scaling():
    base = locking_time(WI_MOON, A_MOON, R_MOON, I_MOON, M_EARTH)
    far = locking_time(WI_MOON, 2 * A_MOON, R_MOON, I_MOON, M_EARTH)
    assert abs(far / base - 2.0 ** 6) < 1e-6, "locking time should scale as a^6"


def test_primary_mass_scaling():
    base = locking_time(WI_MOON, A_MOON, R_MOON, I_MOON, M_EARTH)
    heavy = locking_time(WI_MOON, A_MOON, R_MOON, I_MOON, 2 * M_EARTH)
    assert abs(heavy / base - 0.25) < 1e-9, "locking time should scale as 1/M_p^2"


def test_moon_inside_locking_zone():
    amax = max_locking_distance(WI_MOON, R_MOON, I_MOON, M_EARTH)
    assert amax > A_MOON, "the Moon should sit inside its locking zone"


def test_weaker_primary_smaller_zone():
    strong = max_locking_distance(WI_MOON, R_MOON, I_MOON, M_EARTH)
    weak = max_locking_distance(WI_MOON, R_MOON, I_MOON, M_EARTH / 10.0)
    assert weak < strong, "a weaker primary should give a smaller locking zone"


def test_is_locked_matches_time():
    t = locking_time(WI_MOON, A_MOON, R_MOON, I_MOON, M_EARTH)
    assert is_locked(WI_MOON, A_MOON, R_MOON, I_MOON, M_EARTH,
                     budget=t * 1.01)
    assert not is_locked(WI_MOON, A_MOON, R_MOON, I_MOON, M_EARTH,
                         budget=t * 0.99)


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
