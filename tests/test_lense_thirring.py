"""Frame-dragging / geodetic-precession tests (Gravity Probe B).

Claims checked:
  1. The geodetic (de Sitter) precession at Gravity Probe B's orbit is
     ~6600 mas/yr, matching the measured 6602 mas/yr to ~1%.
  2. The Lense-Thirring frame-dragging precession is ~40 mas/yr, the same order
     as the measured 37.2 mas/yr (and far smaller than geodetic).
  3. Geodetic precession scales as r^{-5/2} and frame-dragging as r^{-3}.
  4. Frame-dragging vanishes for a non-spinning body (J=0).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lense_thirring import (geodetic_rate, frame_dragging_rate,  # noqa: E402
                            geodetic_mas_per_year, frame_dragging_mas_per_year,
                            gravity_probe_b_altitude, M_EARTH, J_EARTH)


def test_geodetic_matches_gpb():
    r = gravity_probe_b_altitude()
    val = geodetic_mas_per_year(M_EARTH, r)
    assert abs(val - 6602.0) / 6602.0 < 0.02, f"geodetic {val} not ~6602 mas/yr"


def test_frame_dragging_matches_gpb():
    r = gravity_probe_b_altitude()
    val = frame_dragging_mas_per_year(J_EARTH, r)
    assert abs(val - 37.2) / 37.2 < 0.15, f"frame-drag {val} not ~37 mas/yr"


def test_geodetic_dominates():
    r = gravity_probe_b_altitude()
    # geodetic is ~160x larger than frame-dragging -- why the latter is hard to see
    assert geodetic_mas_per_year(M_EARTH, r) > 50 * frame_dragging_mas_per_year(J_EARTH, r)


def test_geodetic_r_scaling():
    r = gravity_probe_b_altitude()
    ratio = geodetic_rate(M_EARTH, 2 * r) / geodetic_rate(M_EARTH, r)
    assert abs(ratio - 2 ** (-2.5)) < 1e-9, "geodetic should scale as r^{-5/2}"


def test_frame_dragging_r_scaling():
    r = gravity_probe_b_altitude()
    ratio = frame_dragging_rate(J_EARTH, 2 * r) / frame_dragging_rate(J_EARTH, r)
    assert abs(ratio - 0.125) < 1e-9, "frame-dragging should scale as r^{-3}"


def test_no_spin_no_frame_dragging():
    assert frame_dragging_rate(0.0, gravity_probe_b_altitude()) == 0.0


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
