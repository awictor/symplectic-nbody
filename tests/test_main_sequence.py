"""Main-sequence tests.

Claims checked:
  1. The mass-luminosity relation L ~ M^{3.5}: doubling the mass raises the
     luminosity ~11x, and the Sun is 1 L_sun by construction.
  2. Main-sequence lifetime falls steeply with mass (t ~ M^{-2.5}): the Sun ~10
     Gyr, a 10 M_sun star ~30 Myr, a 0.3 M_sun dwarf outlives the universe.
  3. Effective temperature rises with mass -- massive stars are hot and blue,
     low-mass stars cool and red (the main-sequence slope of the HR diagram).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main_sequence import (luminosity, lifetime_gyr, radius_msun,  # noqa: E402
                           effective_temperature)


def test_mass_luminosity_slope():
    assert abs(luminosity(1.0) - 1.0) < 1e-9, "Sun is 1 L_sun"
    assert abs(luminosity(2.0) / luminosity(1.0) - 2 ** 3.5) < 1e-6


def test_solar_lifetime():
    assert abs(lifetime_gyr(1.0) - 10.0) < 1e-9, "Sun ~10 Gyr"


def test_massive_stars_die_fast():
    assert lifetime_gyr(10.0) < 0.1, "10 M_sun should live < 100 Myr"
    assert lifetime_gyr(30.0) < lifetime_gyr(10.0), "more massive -> shorter"


def test_red_dwarfs_outlive_universe():
    assert lifetime_gyr(0.3) > 13.8, "0.3 M_sun should outlive the universe"


def test_lifetime_scaling():
    # t ~ M^{-2.5}
    r = lifetime_gyr(2.0) / lifetime_gyr(1.0)
    assert abs(r - 2 ** (-2.5)) < 1e-6, "lifetime should scale as M^{-2.5}"


def test_temperature_rises_with_mass():
    assert effective_temperature(10.0) > effective_temperature(1.0) > effective_temperature(0.3)
    # Sun near 5772 K
    assert abs(effective_temperature(1.0) - 5772.0) < 1.0


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
