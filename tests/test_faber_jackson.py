"""Faber-Jackson-relation tests.

Claims checked:
  1. Luminosity scales as sigma^4: doubling the velocity dispersion raises the
     luminosity 16x.
  2. An L* elliptical (sigma ~ 200 km/s) has L ~ 2e10 L_sun, a giant elliptical
     (sigma ~ 300 km/s) ~1e11 L_sun.
  3. The virial mass M ~ sigma^2 R / G is galaxy-scale (~1e10-1e11 M_sun).
  4. Inverting Faber-Jackson recovers the dispersion from the luminosity.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faber_jackson import (faber_jackson_luminosity, virial_mass,  # noqa: E402
                           mass_to_light, dispersion_from_luminosity,
                           M_SUN, KPC)


def test_sigma_fourth_slope():
    r = faber_jackson_luminosity(400e3) / faber_jackson_luminosity(200e3)
    assert abs(r - 16.0) < 1e-6, "L should scale as sigma^4"


def test_lstar_luminosity():
    assert abs(faber_jackson_luminosity(200e3) - 2e10) / 2e10 < 1e-6


def test_giant_elliptical():
    L = faber_jackson_luminosity(300e3)
    assert 8e10 < L < 1.2e11, f"giant elliptical L {L} not ~1e11"


def test_virial_mass_scale():
    M = virial_mass(200e3, 5 * KPC) / M_SUN
    assert 1e10 < M < 1e11, f"virial mass {M} not galaxy-scale"
    # more dispersion or larger radius -> more mass
    assert virial_mass(300e3, 5 * KPC) > virial_mass(200e3, 5 * KPC)


def test_dispersion_inversion():
    for L in (5e9, 2e10, 1e11):
        sigma = dispersion_from_luminosity(L)
        assert abs(faber_jackson_luminosity(sigma) - L) / L < 1e-9


def test_mass_to_light_positive():
    assert mass_to_light(200e3, 5 * KPC) > 0


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
