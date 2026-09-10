"""Atmospheric Jeans-escape tests.

Claims checked:
  1. Earth (exobase ~1000 K) keeps N2, O2, CO2 and water but loses H2 and He.
  2. The Moon, hot and low-gravity, retains essentially nothing (airless).
  3. The escape parameter lambda = G M m / (R k_B T) rises with molecular mass and
     with gravity, and falls with temperature.
  4. The Jeans flux carries an exp(-lambda) Boltzmann factor: heavier gases escape
     exponentially more slowly.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jeans_escape import (escape_speed, thermal_speed, escape_parameter,  # noqa: E402
                          is_retained, jeans_flux, M_EARTH, R_EARTH, M_MOON,
                          R_MOON, M_H2, M_HE, M_N2, M_O2, M_CO2, M_H2O)

T_EXO = 1000.0  # Earth exobase temperature


def test_earth_keeps_heavy_gases():
    for m in (M_N2, M_O2, M_CO2, M_H2O):
        assert is_retained(M_EARTH, R_EARTH, T_EXO, m)


def test_earth_loses_light_gases():
    assert not is_retained(M_EARTH, R_EARTH, T_EXO, M_H2)
    assert not is_retained(M_EARTH, R_EARTH, T_EXO, M_HE)


def test_moon_is_airless():
    # even nitrogen is not retained on the hot, low-gravity Moon
    assert not is_retained(M_MOON, R_MOON, 400.0, M_N2)


def test_escape_speeds():
    assert abs(escape_speed(M_EARTH, R_EARTH) / 1e3 - 11.2) < 0.2
    assert abs(escape_speed(M_MOON, R_MOON) / 1e3 - 2.38) < 0.1


def test_lambda_scalings():
    base = escape_parameter(M_EARTH, R_EARTH, T_EXO, M_N2)
    # heavier molecule -> larger lambda (proportional to m)
    assert abs(escape_parameter(M_EARTH, R_EARTH, T_EXO, 2 * M_N2) / base - 2.0) < 1e-9
    # hotter exobase -> smaller lambda (1/T)
    assert abs(escape_parameter(M_EARTH, R_EARTH, 2 * T_EXO, M_N2) / base - 0.5) < 1e-9


def test_lambda_is_vesc_over_vth_squared():
    lam = escape_parameter(M_EARTH, R_EARTH, T_EXO, M_N2)
    ve = escape_speed(M_EARTH, R_EARTH)
    vth = thermal_speed(T_EXO, M_N2)
    assert abs(lam - (ve / vth) ** 2) < 1e-6


def test_flux_heavier_is_smaller():
    # at fixed n, T, world: a heavier gas escapes far more slowly
    fh2 = jeans_flux(1e13, T_EXO, M_H2, M_EARTH, R_EARTH)
    fn2 = jeans_flux(1e13, T_EXO, M_N2, M_EARTH, R_EARTH)
    assert fh2 > fn2
    assert fn2 / fh2 < 1e-20, "N2 escape should be astronomically slower than H2"


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
