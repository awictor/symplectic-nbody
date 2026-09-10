"""Magnetic-braking / gyrochronology tests.

Claims checked:
  1. Skumanich: rotation rate ~ t^(-1/2), period ~ t^(1/2).
  2. Gyrochronology inverts to the Sun: a 25.4-day period gives ~4.6 Gyr; a
     10-day rotator is young (~0.7 Gyr).
  3. The Alfven radius of the solar wind is many stellar radii (long lever arm).
  4. The Weber-Davis torque grows with rotation and with r_A^2.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from magnetic_braking import (rotation_rate, rotation_period, gyro_age,  # noqa: E402
                              wind_torque, spin_down_time, alfven_radius,
                              P_SUN, T_SUN, OMEGA_SUN, DAY, GYR)

R_SUN = 6.957e8


def test_skumanich_period_grows_as_sqrt_t():
    # quadruple the age -> double the period
    assert abs(rotation_period(4 * T_SUN) / rotation_period(T_SUN) - 2.0) < 1e-9


def test_rate_falls_as_inverse_sqrt_t():
    assert abs(rotation_rate(4 * T_SUN) / rotation_rate(T_SUN) - 0.5) < 1e-9


def test_period_at_solar_age():
    p = rotation_period(T_SUN) / DAY
    assert abs(p - 25.4) < 0.1, f"solar-age period {p} d not ~25.4"


def test_gyro_age_recovers_sun():
    age = gyro_age(P_SUN) / GYR
    assert abs(age - 4.567) < 0.05, f"gyro age of Sun {age} Gyr not ~4.6"


def test_fast_rotator_is_young():
    age = gyro_age(10 * DAY) / GYR
    assert age < 1.0, f"a 10-day rotator should be young, got {age} Gyr"


def test_alfven_radius_is_many_radii():
    rA = alfven_radius(2e-4, R_SUN, 2e9, 4e5) / R_SUN
    assert 5.0 < rA < 30.0, f"solar r_A {rA} R_sun out of expected range"


def test_torque_scales_with_lever_arm():
    t1 = wind_torque(2e9, OMEGA_SUN, 5 * R_SUN)
    t2 = wind_torque(2e9, OMEGA_SUN, 10 * R_SUN)
    assert abs(t2 / t1 - 4.0) < 1e-9, "torque should scale as r_A^2"


def test_torque_scales_with_rotation():
    t1 = wind_torque(2e9, OMEGA_SUN, 8 * R_SUN)
    t2 = wind_torque(2e9, 2 * OMEGA_SUN, 8 * R_SUN)
    assert abs(t2 / t1 - 2.0) < 1e-9


def test_spin_down_time_gyr_scale():
    rA = alfven_radius(2e-4, R_SUN, 2e9, 4e5)
    tau = spin_down_time(7e46, OMEGA_SUN, 2e9, rA) / GYR
    assert 1.0 < tau < 100.0, f"spin-down time {tau} Gyr off order of magnitude"


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
