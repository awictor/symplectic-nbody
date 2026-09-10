"""Parallax / proper-motion / space-velocity tests.

Claims checked:
  1. The parsec definition: 1 arcsecond of parallax = 1 pc; Proxima (0.7687") is
     1.30 pc = 4.24 ly.
  2. Distance is inverse in parallax; parallax and distance invert cleanly.
  3. v_t = 4.74 mu d reproduces Barnard's Star's ~90 km/s tangential velocity, and
     the space velocity combines radial and tangential in quadrature.
  4. Proper motion from a tangential velocity round-trips.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parallax import (distance_pc, distance_ly, parallax_at,  # noqa: E402
                      tangential_velocity, space_velocity,
                      proper_motion_from_vt)


def test_parsec_definition():
    assert abs(distance_pc(1.0) - 1.0) < 1e-12


def test_proxima():
    d = distance_pc(0.7687)
    assert abs(d - 1.301) < 0.005
    assert abs(distance_ly(0.7687) - 4.24) < 0.05


def test_inverse_relation():
    assert abs(distance_pc(0.1) - 10.0) < 1e-9
    assert abs(parallax_at(10.0) - 0.1) < 1e-12


def test_parallax_distance_invert():
    for d in (1.3, 10.0, 100.0):
        assert abs(distance_pc(parallax_at(d)) - d) < 1e-9


def test_barnard_tangential():
    v_t = tangential_velocity(10.36, 1.83)
    assert abs(v_t - 90.0) < 2.0, f"Barnard tangential {v_t} km/s not ~90"


def test_space_velocity_quadrature():
    v = space_velocity(-110.0, 10.36, 1.83)
    v_t = tangential_velocity(10.36, 1.83)
    assert abs(v - math.hypot(110.0, v_t)) < 1e-9
    assert v > v_t  # adding radial motion increases the total


def test_proper_motion_roundtrip():
    v_t = tangential_velocity(5.0, 2.0)
    assert abs(proper_motion_from_vt(v_t, 2.0) - 5.0) < 1e-9


def test_farther_star_smaller_parallax():
    assert parallax_at(100.0) < parallax_at(10.0)


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
