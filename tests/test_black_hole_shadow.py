"""Black-hole-shadow tests.

Claims checked:
  1. The photon sphere is 1.5 r_s and the shadow diameter is 3 sqrt(3) ~ 5.196 r_s --
     bigger than the ~2 r_s horizon because of light bending.
  2. M87* (6.5e9 Msun, 16.8 Mpc) casts a ~40 microarcsecond shadow, and Sgr A*
     (4.15e6 Msun, 8.15 kpc) a ~52 uas shadow -- the EHT measurements.
  3. The shadow angular size scales as M and as 1/D.
  4. b_crit = 3 sqrt(3) G M / c^2.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from black_hole_shadow import (schwarzschild_radius, photon_sphere_radius,  # noqa: E402
                               critical_impact_parameter, shadow_diameter,
                               shadow_angular_diameter, shadow_angular_diameter_uas,
                               shadow_in_rs, M_SUN, PC, MPC, G, C)


def test_photon_sphere_is_1p5_rs():
    M = 10 * M_SUN
    assert abs(photon_sphere_radius(M) / schwarzschild_radius(M) - 1.5) < 1e-12


def test_shadow_is_3sqrt3_rs():
    assert abs(shadow_in_rs(M_SUN) - 3.0 * math.sqrt(3.0)) < 1e-9


def test_m87_shadow():
    theta = shadow_angular_diameter_uas(6.5e9 * M_SUN, 16.8 * MPC)
    assert abs(theta - 40.0) < 3.0, f"M87* shadow {theta} uas not ~40"


def test_sgr_a_shadow():
    theta = shadow_angular_diameter_uas(4.15e6 * M_SUN, 8150 * PC)
    assert abs(theta - 52.0) < 3.0, f"Sgr A* shadow {theta} uas not ~52"


def test_bcrit_formula():
    M = 5 * M_SUN
    assert abs(critical_impact_parameter(M) - 3.0 * math.sqrt(3.0) * G * M / C ** 2) < 1e-6


def test_scales_with_mass():
    D = 10 * MPC
    assert abs(shadow_angular_diameter(2e9 * M_SUN, D) / shadow_angular_diameter(1e9 * M_SUN, D) - 2.0) < 1e-9


def test_scales_inverse_distance():
    M = 1e9 * M_SUN
    assert abs(shadow_angular_diameter(M, 20 * MPC) / shadow_angular_diameter(M, 10 * MPC) - 0.5) < 1e-9


def test_shadow_bigger_than_horizon():
    # the shadow (5.196 r_s across) is larger than the 2 r_s horizon diameter
    assert shadow_in_rs(M_SUN) > 2.0


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
