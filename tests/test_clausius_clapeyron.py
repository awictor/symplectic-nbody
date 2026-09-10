"""Clausius-Clapeyron tests.

Claims checked:
  1. Water's vapor pressure is 1 atm at 100 C by construction, and boiling point drops
     with ambient pressure (Everest ~72 C, pressure cooker ~121 C at 2 atm).
  2. Vapor pressure rises steeply (exponentially) with temperature.
  3. The latent heat is recovered from two coexistence points.
  4. Boiling point and vapor pressure invert consistently.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clausius_clapeyron import (vapor_pressure, boiling_point,  # noqa: E402
                                latent_heat_from_two_points,
                                fractional_pressure_change,
                                pressure_from_altitude, L_WATER, T_BOIL_WATER,
                                P_ATM)


def test_boils_at_100C():
    assert abs(vapor_pressure(373.15) - P_ATM) < 1.0


def test_everest_boiling():
    P = pressure_from_altitude(8848.0)
    Tb = boiling_point(P) - 273.15
    assert 65.0 < Tb < 80.0, f"Everest boiling {Tb} C off scale"


def test_pressure_cooker():
    Tb = boiling_point(2.0 * P_ATM) - 273.15
    assert 115.0 < Tb < 125.0, f"2-atm boiling {Tb} C off scale"


def test_vapor_pressure_rises_steeply():
    # roughly doubles over ~15 K near room temperature
    r = vapor_pressure(308.0) / vapor_pressure(293.0)
    assert r > 1.8


def test_latent_heat_recovery():
    L = latent_heat_from_two_points(vapor_pressure(350.0), 350.0,
                                    vapor_pressure(373.15), 373.15)
    assert abs(L - L_WATER) < 1.0


def test_boiling_vapor_invert():
    for P in (0.5 * P_ATM, P_ATM, 2.0 * P_ATM):
        Tb = boiling_point(P)
        assert abs(vapor_pressure(Tb) - P) < 1.0


def test_lower_pressure_lower_boiling():
    assert boiling_point(0.5 * P_ATM) < boiling_point(P_ATM) < boiling_point(2.0 * P_ATM)


def test_fractional_change_positive():
    assert fractional_pressure_change(300.0) > 0.0
    # (1/P)dP/dT ~ L/RT^2 ~ 0.05 /K near room temp
    assert abs(fractional_pressure_change(300.0) - L_WATER / (8.314462618 * 300.0 ** 2)) < 1e-9


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
