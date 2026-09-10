"""Rossby-number / geostrophic-balance tests.

Claims checked:
  1. The Coriolis parameter f = 2 Omega sin(lat) vanishes at the equator and peaks
     at the poles; f(45 deg) ~ 1e-4 s^-1.
  2. A synoptic weather system (U~10 m/s, L~1000 km) is geostrophic (Ro < 0.1); a
     tornado (small L) is not.
  3. The geostrophic wind U_g = (1/rho f) dp/dn gives ~10 m/s for typical isobar
     spacing.
  4. The inertial period 2 pi / f is ~17 hr at mid-latitudes (half a pendulum day).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rossby import (coriolis_parameter, rossby_number, is_geostrophic,  # noqa: E402
                    geostrophic_wind, deformation_radius, inertial_period,
                    OMEGA_EARTH)


def test_coriolis_equator_and_pole():
    assert abs(coriolis_parameter(0.0)) < 1e-12
    fp = coriolis_parameter(90.0)
    assert abs(fp - 2.0 * OMEGA_EARTH) < 1e-12


def test_f_at_45():
    f = coriolis_parameter(45.0)
    assert abs(f - 1.03e-4) < 5e-6, f"f(45) {f} not ~1e-4"


def test_cyclone_is_geostrophic():
    f = coriolis_parameter(45.0)
    assert is_geostrophic(10.0, 1e6, f)
    assert rossby_number(10.0, 1e6, f) < 0.1


def test_tornado_not_geostrophic():
    f = coriolis_parameter(45.0)
    assert not is_geostrophic(100.0, 100.0, f)
    assert rossby_number(100.0, 100.0, f) > 100.0


def test_geostrophic_wind_scale():
    f = coriolis_parameter(45.0)
    U = geostrophic_wind(1e-3, 1.2, f)   # ~1 mb / 100 km
    assert 3.0 < U < 20.0, f"geostrophic wind {U} m/s off scale"


def test_southern_hemisphere_sign():
    assert coriolis_parameter(-30.0) < 0.0 < coriolis_parameter(30.0)


def test_inertial_period_midlatitude():
    f = coriolis_parameter(45.0)
    hr = inertial_period(f) / 3600.0
    assert 15.0 < hr < 19.0, f"inertial period {hr} hr not ~17"


def test_deformation_radius_scale():
    f = coriolis_parameter(45.0)
    L = deformation_radius(30.0, f) / 1e3   # km
    assert 100.0 < L < 1000.0, f"deformation radius {L} km off scale"


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
