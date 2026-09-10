"""Relativistic-beaming tests.

Claims checked:
  1. The beaming cone half-angle is ~1/gamma (5.7 deg for gamma=10).
  2. An approaching jet within the cone is boosted (D>1, large flux boost); the
     receding counter-jet is dimmed (D<1), giving an enormous jet/counter-jet ratio.
  3. Apparent transverse motion can be superluminal, peaking near gamma*beta at
     cos theta = beta.
  4. The Doppler factor is 1 at the aberration angle and the boost powers are correct.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beaming import (lorentz_factor, doppler_factor, flux_boost,  # noqa: E402
                     beaming_cone_halfangle, jet_counterjet_ratio,
                     apparent_transverse_speed)

BETA10 = math.sqrt(1.0 - 1.0 / 100.0)   # gamma = 10


def test_lorentz_factor():
    assert abs(lorentz_factor(BETA10) - 10.0) < 1e-6


def test_cone_halfangle():
    a = math.degrees(beaming_cone_halfangle(BETA10))
    assert abs(a - 5.73) < 0.1, f"beaming cone {a} deg not ~1/gamma"


def test_approaching_boosted():
    D = doppler_factor(BETA10, math.radians(10.0))
    assert D > 1.0
    assert flux_boost(BETA10, math.radians(10.0)) > 100.0


def test_receding_dimmed():
    D = doppler_factor(BETA10, math.radians(170.0))
    assert D < 1.0


def test_huge_jet_ratio():
    R = jet_counterjet_ratio(BETA10, math.radians(10.0))
    assert R > 1e5, f"jet/counter-jet ratio {R} should be enormous"


def test_superluminal():
    v_app = apparent_transverse_speed(BETA10, math.radians(10.0))
    assert v_app > 1.0, "apparent transverse speed should exceed c"


def test_max_apparent_near_gamma_beta():
    # maximum apparent speed at cos theta = beta is gamma*beta
    theta_opt = math.acos(BETA10)
    v_max = apparent_transverse_speed(BETA10, theta_opt)
    assert abs(v_max - lorentz_factor(BETA10) * BETA10) < 0.1


def test_continuous_lower_exponent():
    # a steady jet boosts by D^(2+alpha), less than a blob's D^(3+alpha)
    blob = flux_boost(BETA10, math.radians(10.0), continuous=False)
    jet = flux_boost(BETA10, math.radians(10.0), continuous=True)
    assert jet < blob


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
