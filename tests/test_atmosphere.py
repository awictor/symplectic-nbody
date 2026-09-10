"""Atmospheric-escape (Jeans) tests.

Claims checked:
  1. Earth keeps heavy gases (N2, O2) but loses light ones (H2, He) from its
     ~1000 K exosphere -- exactly the composition we observe.
  2. The Moon retains essentially nothing.
  3. Jupiter retains even hydrogen.
  4. The escape parameter lambda = v_esc^2 / v_th^2 scales with molecular mass
     (heavier is retained more easily) and inversely with temperature.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atmosphere import (thermal_speed, escape_speed, escape_parameter,  # noqa: E402
                        is_retained)

M_EARTH, R_EARTH = 5.972e24, 6.371e6
M_MOON, R_MOON = 7.342e22, 1.737e6
M_JUP, R_JUP = 1.898e27, 6.991e7


def test_earth_keeps_heavy_loses_light():
    T = 1000.0
    assert not is_retained(M_EARTH, R_EARTH, T, 2), "Earth should lose H2"
    assert not is_retained(M_EARTH, R_EARTH, T, 4), "Earth should lose He"
    assert is_retained(M_EARTH, R_EARTH, T, 28), "Earth should keep N2"
    assert is_retained(M_EARTH, R_EARTH, T, 32), "Earth should keep O2"


def test_moon_keeps_nothing():
    # hot dayside subsolar exosphere (~day-side temperatures); even N2 escapes,
    # and lighter gases far more so -- the Moon holds no appreciable atmosphere
    assert not is_retained(M_MOON, R_MOON, 400.0, 28), "Moon should lose N2"
    assert not is_retained(M_MOON, R_MOON, 400.0, 4), "Moon should lose He"


def test_jupiter_keeps_hydrogen():
    assert is_retained(M_JUP, R_JUP, 1000.0, 2), "Jupiter should keep H2"


def test_heavier_retained_more_easily():
    lam_light = escape_parameter(M_EARTH, R_EARTH, 1000.0, 2)
    lam_heavy = escape_parameter(M_EARTH, R_EARTH, 1000.0, 28)
    assert lam_heavy > lam_light, "heavier molecules have larger lambda"
    # lambda linear in molecular mass
    assert abs(escape_parameter(M_EARTH, R_EARTH, 1000.0, 4)
               / escape_parameter(M_EARTH, R_EARTH, 1000.0, 2) - 2.0) < 1e-6


def test_hotter_lowers_lambda():
    hot = escape_parameter(M_EARTH, R_EARTH, 2000.0, 28)
    cool = escape_parameter(M_EARTH, R_EARTH, 1000.0, 28)
    assert hot < cool, "higher temperature lowers the escape parameter"


def test_thermal_speed_lighter_faster():
    assert thermal_speed(300.0, 2) > thermal_speed(300.0, 28)


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
