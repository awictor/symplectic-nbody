"""Gravitational-time tests: redshift, GPS, and the Shapiro delay.

Claims checked:
  1. Pound-Rebka: over a 22.5 m tower the redshift is ~2.5e-15.
  2. GPS satellites gain ~+38 microseconds/day relative to the ground (GR clock
     speed-up beats SR time dilation) -- the correction without which GPS fails.
  3. The Sun's surface gravitational redshift is z = GM/(Rc^2) ~ 2.1e-6.
  4. The Shapiro round-trip delay for radar grazing the Sun is a few hundred
     microseconds, matching the Cassini / Earth-Venus measurements.
  5. The exact Schwarzschild redshift reduces to g h / c^2 in the weak field.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gr_time import (gravitational_redshift, redshift_uniform_field,  # noqa: E402
                     gps_time_gain_per_day, shapiro_delay,
                     M_SUN, R_SUN, R_EARTH, M_EARTH, AU, G, C)


def test_pound_rebka():
    z = redshift_uniform_field(9.8, 22.5)
    assert abs(z - 2.5e-15) < 0.2e-15, f"Pound-Rebka z {z} not ~2.5e-15"


def test_gps_clock_gain():
    us = gps_time_gain_per_day() * 1e6
    assert 35.0 < us < 40.0, f"GPS clock gain {us} us/day not ~38"


def test_sun_surface_redshift():
    z = gravitational_redshift(M_SUN, R_SUN, 1e15)
    assert abs(z - 2.12e-6) / 2.12e-6 < 0.02, f"Sun redshift {z} not ~2.12e-6"


def test_shapiro_delay_scale():
    dt = shapiro_delay(AU, 8.5 * AU, R_SUN)
    assert 200e-6 < dt < 320e-6, f"Shapiro delay {dt*1e6} us off scale"


def test_shapiro_scales_with_mass():
    d1 = shapiro_delay(AU, AU, R_SUN, M=M_SUN)
    d2 = shapiro_delay(AU, AU, R_SUN, M=2 * M_SUN)
    assert abs(d2 / d1 - 2.0) < 1e-9, "Shapiro delay linear in mass"


def test_weak_field_reduction():
    # exact Schwarzschild redshift over a small height h at Earth's surface
    # should match g h / c^2 with g = GM/R^2
    h = 100.0
    z_exact = gravitational_redshift(M_EARTH, R_EARTH, R_EARTH + h)
    g = G * M_EARTH / R_EARTH ** 2
    z_weak = redshift_uniform_field(g, h)
    assert abs(z_exact - z_weak) / z_weak < 1e-2, "exact redshift should match g h/c^2"


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
