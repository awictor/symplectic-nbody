"""Eddington-luminosity tests.

Claims checked:
  1. L_Edd = 4 pi G M m_p c / sigma_T is linear in mass and equals ~1.26e31 W
     (~3.3e4 solar luminosities) for one solar mass.
  2. The Salpeter e-folding time is ~45 Myr, independent of mass.
  3. Growing a 10-solar-mass seed to a billion-solar-mass quasar takes ~0.8 Gyr
     of Eddington-limited accretion -- just fitting inside the age of the early
     universe, which is why the first quasars are a puzzle.
  4. The maximum accretion rate scales with mass and with 1/efficiency.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eddington import (eddington_luminosity, eddington_luminosity_solar_units,  # noqa: E402
                       eddington_accretion_rate, salpeter_time, growth_time,
                       M_SUN, YEAR)


def test_solar_eddington_luminosity():
    L = eddington_luminosity(M_SUN)
    assert abs(L - 1.26e31) / 1.26e31 < 0.02, f"solar L_Edd {L} not ~1.26e31 W"
    Lsun = eddington_luminosity_solar_units(1.0)
    assert 3.0e4 < Lsun < 3.6e4, f"solar L_Edd {Lsun} not ~3.3e4 L_sun"


def test_linear_in_mass():
    assert abs(eddington_luminosity(3 * M_SUN) / eddington_luminosity(M_SUN) - 3.0) < 1e-12


def test_salpeter_time_is_45_myr():
    t = salpeter_time() / YEAR / 1e6
    assert 40.0 < t < 50.0, f"Salpeter time {t} Myr not ~45"


def test_salpeter_time_independent_of_mass():
    # growth_time depends only on the mass ratio, not absolute mass
    t1 = growth_time(M_SUN, 100 * M_SUN)
    t2 = growth_time(1000 * M_SUN, 100000 * M_SUN)  # same ratio 100
    assert abs(t1 - t2) / t1 < 1e-9


def test_quasar_growth_time():
    t = growth_time(10 * M_SUN, 1e9 * M_SUN) / YEAR / 1e6
    assert 700.0 < t < 950.0, f"seed->quasar growth {t} Myr off"


def test_accretion_rate_scaling():
    m1 = eddington_accretion_rate(M_SUN, efficiency=0.1)
    m2 = eddington_accretion_rate(2 * M_SUN, efficiency=0.1)
    assert abs(m2 / m1 - 2.0) < 1e-9, "accretion rate linear in mass"
    # lower efficiency -> higher required rate for the same luminosity
    assert eddington_accretion_rate(M_SUN, 0.05) > eddington_accretion_rate(M_SUN, 0.1)


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
