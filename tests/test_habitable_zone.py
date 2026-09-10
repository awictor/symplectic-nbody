"""Habitable-zone tests.

Claims checked:
  1. Earth's equilibrium temperature is ~255 K (the greenhouse effect then warms
     the surface to 288 K); zero albedo gives ~279 K.
  2. Equilibrium temperature scales as L^{1/4} / sqrt(d).
  3. The habitable-zone distance scales as sqrt(L_star): a 4x-more-luminous star
     has its HZ twice as far out; a dim red dwarf's HZ is tucked in close.
  4. The inner edge is hotter/closer than the outer edge.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from habitable_zone import (equilibrium_temperature, habitable_zone,  # noqa: E402
                            hz_center)


def test_earth_equilibrium_temperature():
    T = equilibrium_temperature(1.0, 1.0, 0.3)
    assert abs(T - 255.0) < 5.0, f"Earth T_eq {T} not ~255 K"


def test_zero_albedo_temperature():
    assert abs(equilibrium_temperature(1.0, 1.0, 0.0) - 278.5) < 1.0


def test_temperature_scalings():
    base = equilibrium_temperature(1.0, 1.0)
    # T ~ L^{1/4}
    assert abs(equilibrium_temperature(16.0, 1.0) / base - 2.0) < 1e-6
    # T ~ 1/sqrt(d)
    assert abs(equilibrium_temperature(1.0, 4.0) / base - 0.5) < 1e-6


def test_hz_scales_as_sqrt_luminosity():
    assert abs(hz_center(4.0) / hz_center(1.0) - 2.0) < 1e-6
    assert abs(hz_center(100.0) / hz_center(1.0) - 10.0) < 1e-6


def test_red_dwarf_hz_close_in():
    assert hz_center(0.01) < 0.2, "a 0.01 L_sun star's HZ should be tucked in close"


def test_inner_closer_than_outer():
    inner, outer = habitable_zone(1.0)
    assert inner < outer, "inner (hotter) edge should be closer than outer"


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
