"""Exoplanet-detection tests (transit + radial velocity).

Claims checked:
  1. Transit depth (R_p/R_star)^2: Jupiter dims the Sun ~1%, Earth ~8.4e-5.
  2. The radial-velocity semi-amplitude: Jupiter wobbles the Sun ~12.5 m/s,
     Earth only ~0.09 m/s.
  3. Kepler's period comes out right (Jupiter ~11.9 yr).
  4. A hot Jupiter (0.05 AU) produces a large RV signal and a short period,
     which is why the first exoplanets found were hot Jupiters.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exoplanet import (transit_depth, orbital_period, transit_duration,  # noqa: E402
                       rv_semi_amplitude, M_SUN, R_SUN, M_JUP, R_JUP,
                       M_EARTH, R_EARTH, AU, YEAR)


def test_jupiter_transit_depth():
    assert abs(transit_depth(R_JUP, R_SUN) - 0.01) < 0.002


def test_earth_transit_depth():
    assert abs(transit_depth(R_EARTH, R_SUN) - 8.4e-5) / 8.4e-5 < 0.05


def test_jupiter_rv_amplitude():
    K = rv_semi_amplitude(M_JUP, M_SUN, 5.204 * AU)
    assert abs(K - 12.5) < 0.5, f"Jupiter RV {K} m/s not ~12.5"


def test_earth_rv_amplitude():
    K = rv_semi_amplitude(M_EARTH, M_SUN, AU)
    assert abs(K - 0.09) < 0.02, f"Earth RV {K} m/s not ~0.09"


def test_jupiter_period():
    assert abs(orbital_period(5.204 * AU, M_SUN) / YEAR - 11.9) < 0.2


def test_hot_jupiter_signals():
    K = rv_semi_amplitude(M_JUP, M_SUN, 0.05 * AU)
    P_days = orbital_period(0.05 * AU, M_SUN) / 86400.0
    assert K > 50.0, f"hot Jupiter RV {K} m/s should be large"
    assert P_days < 10.0, f"hot Jupiter period {P_days} d should be short"
    # closer-in planet has a stronger RV signal
    assert rv_semi_amplitude(M_JUP, M_SUN, 0.05 * AU) > rv_semi_amplitude(M_JUP, M_SUN, 5.2 * AU)


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
