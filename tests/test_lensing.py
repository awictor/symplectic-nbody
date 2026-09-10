"""Gravitational-lensing tests.

Claims checked:
  1. The GR deflection of starlight at the Sun's limb is ~1.75 arcsec (the 1919
     Eddington result), and it scales as M/b.
  2. Both image positions solve the point-mass lens equation
     beta = theta - theta_E^2/theta exactly.
  3. A perfectly aligned source (beta=0) produces an Einstein ring at theta_E
     (both images at +/- theta_E).
  4. The summed magnification of the two images equals the closed-form total
     magnification A(u) = (u^2+2)/(u sqrt(u^2+4)).
  5. Total magnification rises as the source approaches alignment (u -> 0).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import (deflection_angle, image_positions, magnification,  # noqa: E402
                     total_magnification, einstein_radius)

M_SUN = 1.989e30
R_SUN = 6.96e8


def test_sun_limb_deflection():
    alpha_arcsec = math.degrees(deflection_angle(M_SUN, R_SUN)) * 3600.0
    assert abs(alpha_arcsec - 1.75) < 0.02, f"Sun deflection {alpha_arcsec} not ~1.75 arcsec"


def test_deflection_scaling():
    base = deflection_angle(M_SUN, R_SUN)
    assert abs(deflection_angle(2 * M_SUN, R_SUN) - 2 * base) < 1e-30 + 1e-6 * base
    assert abs(deflection_angle(M_SUN, 2 * R_SUN) - 0.5 * base) < 1e-6 * base


def test_images_solve_lens_equation():
    tE = 1.0
    for beta in (0.0, 0.3, 1.0, 2.5):
        tp, tm = image_positions(beta, tE)
        for th in (tp, tm):
            assert abs((th - tE * tE / th) - beta) < 1e-12, f"image {th} fails lens eq at beta={beta}"


def test_einstein_ring_at_perfect_alignment():
    tE = 2.3
    tp, tm = image_positions(0.0, tE)
    assert abs(tp - tE) < 1e-12 and abs(tm + tE) < 1e-12, "beta=0 should give +/- theta_E ring"


def test_magnification_sum_matches_formula():
    tE = 1.0
    for beta in (0.2, 0.5, 1.0, 2.0):
        tp, tm = image_positions(beta, tE)
        summed = abs(magnification(tp, tE)) + abs(magnification(tm, tE))
        assert abs(summed - total_magnification(beta, tE)) < 1e-9, f"mag sum mismatch at beta={beta}"


def test_magnification_rises_toward_alignment():
    tE = 1.0
    far = total_magnification(2.0, tE)
    near = total_magnification(0.2, tE)
    assert near > far > 1.0, f"closer alignment should magnify more: {near} vs {far}"


def test_einstein_radius_positive():
    tE = einstein_radius(M_SUN, 1e19, 2e19, 1e19)
    assert tE > 0


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
