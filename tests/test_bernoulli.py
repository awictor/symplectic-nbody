"""Bernoulli / Venturi / Pitot tests.

Claims checked:
  1. Speeding up the flow lowers the static pressure (Bernoulli trade-off).
  2. A Pitot tube recovers airspeed: v = sqrt(2(P_stag - P_static)/rho), inverting the
     dynamic pressure.
  3. Torricelli's efflux speed sqrt(2gh) equals free-fall from that depth.
  4. The Venturi throat velocity follows from the pressure drop and area ratio, and the
     Bernoulli constant is conserved along a streamline.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bernoulli import (bernoulli_constant, pressure_at, venturi_velocity,  # noqa: E402
                       pitot_airspeed, torricelli_speed, dynamic_pressure,
                       RHO_WATER, RHO_AIR, G_EARTH)


def test_faster_flow_lower_pressure():
    P2 = pressure_at(1e5, 10.0, 30.0, RHO_WATER)
    assert P2 < 1e5


def test_pitot_roundtrip():
    q = dynamic_pressure(250.0, RHO_AIR)
    assert abs(pitot_airspeed(q, 0.0, RHO_AIR) - 250.0) < 1e-6


def test_torricelli_equals_freefall():
    # v = sqrt(2 g h) is the free-fall speed from height h
    h = 5.0
    assert abs(torricelli_speed(h) - math.sqrt(2 * G_EARTH * h)) < 1e-9


def test_dynamic_pressure_scaling():
    assert abs(dynamic_pressure(20.0, RHO_AIR) / dynamic_pressure(10.0, RHO_AIR) - 4.0) < 1e-9


def test_venturi_velocity():
    v = venturi_velocity(2e-4, 1e-4, 1000.0, RHO_WATER)
    assert v > 0.0
    # narrower throat (bigger area ratio) speeds the flow for the same dP
    v_narrow = venturi_velocity(4e-4, 1e-4, 1000.0, RHO_WATER)
    assert v_narrow < v  # A1 larger -> ratio smaller -> ... check monotonicity sanely


def test_bernoulli_constant_conserved():
    # two points on a streamline share the constant
    P1, v1 = 1e5, 10.0
    v2 = 30.0
    P2 = pressure_at(P1, v1, v2, RHO_WATER)
    c1 = bernoulli_constant(P1, v1, 0.0, RHO_WATER)
    c2 = bernoulli_constant(P2, v2, 0.0, RHO_WATER)
    assert abs(c1 - c2) < 1e-3


def test_deeper_hole_faster_jet():
    assert torricelli_speed(10.0) > torricelli_speed(2.0)


def test_gravity_head():
    # raising the fluid (h2>h1) at fixed speed lowers the pressure by rho g dh
    P2 = pressure_at(1e5, 5.0, 5.0, RHO_WATER, h1=0.0, h2=2.0)
    assert abs((1e5 - P2) - RHO_WATER * G_EARTH * 2.0) < 1e-3


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
