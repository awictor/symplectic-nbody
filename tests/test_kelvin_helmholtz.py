"""Kelvin-Helmholtz-timescale tests.

Claims checked:
  1. The Sun's Kelvin-Helmholtz (thermal) time is ~30 Myr.
  2. That is hundreds of times shorter than the Earth's age -- the historic proof
     that the Sun cannot be powered by gravitational contraction alone (it needs
     nuclear fusion, whose main-sequence lifetime is ~10 Gyr).
  3. The KH time scales as M^2 / (R L); a more luminous star relaxes faster.
  4. The gravitational binding energy is ~ G M^2 / R.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kelvin_helmholtz import (kelvin_helmholtz_time, solar_kh_time_myr,  # noqa: E402
                              kh_time_solar_units,
                              gravitational_binding_energy,
                              M_SUN, R_SUN, L_SUN)


def test_solar_kh_time():
    t = solar_kh_time_myr()
    assert 20 < t < 45, f"solar KH time {t} Myr not ~30"


def test_kh_far_short_of_earth_age():
    # Earth is ~4.5 Gyr; the KH time is >100x shorter
    assert 4500.0 / solar_kh_time_myr() > 100, "KH time should be << Earth's age"


def test_kh_much_shorter_than_nuclear():
    # nuclear main-sequence lifetime ~10 Gyr = 1e4 Myr
    assert solar_kh_time_myr() < 0.01 * 1e4, "KH time should be << nuclear lifetime"


def test_luminosity_scaling():
    # brighter star relaxes faster: t_KH ~ 1/L
    t1 = kh_time_solar_units(1.0, 1.0, 1.0)
    t2 = kh_time_solar_units(1.0, 1.0, 2.0)
    assert abs(t1 / t2 - 2.0) < 1e-9, "t_KH ~ 1/L"


def test_mass_squared_over_radius():
    # t_KH ~ M^2 / R at fixed L
    base = kh_time_solar_units(1.0, 1.0, 1.0)
    assert abs(kh_time_solar_units(2.0, 1.0, 1.0) / base - 4.0) < 1e-9, "t_KH ~ M^2"
    assert abs(kh_time_solar_units(1.0, 2.0, 1.0) / base - 0.5) < 1e-9, "t_KH ~ 1/R"


def test_binding_energy():
    E = gravitational_binding_energy(M_SUN, R_SUN)
    assert 1e41 < E < 1e42, f"solar binding energy {E} J off scale"


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
