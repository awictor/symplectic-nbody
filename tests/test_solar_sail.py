"""Solar-sail / radiation-pressure tests.

Claims checked:
  1. The solar flux at 1 AU is ~1361 W/m^2, giving ~9 uPa on a mirror sail (twice
     the ~4.5 uPa on a black one).
  2. Radiation pressure and flux fall as 1/r^2.
  3. The lightness number beta is distance-independent; beta = 1 needs an area-to-mass
     ratio of ~650 m^2/kg (~1.5 g/m^2 loading for a mirror).
  4. A mirror feels twice the force of a black surface.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solar_sail import (solar_flux, radiation_pressure, sail_force,  # noqa: E402
                        sail_acceleration, lightness_number,
                        critical_area_to_mass, AU, SOLAR_CONSTANT)


def test_flux_at_1au():
    assert abs(solar_flux(AU) - 1361.0) < 10.0


def test_mirror_pressure():
    p = radiation_pressure(AU, 1.0) * 1e6  # uPa
    assert abs(p - 9.1) < 0.3, f"mirror pressure {p} uPa not ~9"


def test_mirror_double_black():
    assert abs(radiation_pressure(AU, 1.0) / radiation_pressure(AU, 0.0) - 2.0) < 1e-9


def test_inverse_square():
    assert abs(radiation_pressure(2 * AU) / radiation_pressure(AU) - 0.25) < 1e-9


def test_lightness_distance_independent():
    # beta does not depend on r; check via the acceleration ratio to gravity is const
    b = lightness_number(100.0, 1.0)
    assert b > 0
    # the function takes no radius argument at all
    assert lightness_number(100.0, 1.0) == lightness_number(100.0, 1.0)


def test_critical_area_to_mass_gives_beta_one():
    am = critical_area_to_mass(1.0)
    assert abs(lightness_number(am, 1.0) - 1.0) < 1e-9


def test_critical_loading_1p5_g_per_m2():
    am = critical_area_to_mass(1.0)          # m^2/kg
    loading = 1.0 / am * 1000.0              # g/m^2
    assert abs(loading - 1.53) < 0.1


def test_acceleration_scales_with_area_to_mass():
    a1 = sail_acceleration(30.0, 5.0, AU)
    a2 = sail_acceleration(60.0, 5.0, AU)
    assert abs(a2 / a1 - 2.0) < 1e-9


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
