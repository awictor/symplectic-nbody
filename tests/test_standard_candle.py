"""Standard-candle / distance-modulus tests.

Claims checked:
  1. The distance modulus is 0 at 10 pc and ~18.5 for the LMC (50 kpc); it inverts
     cleanly to distance.
  2. Five magnitudes is exactly a factor of 100 in flux.
  3. The Cepheid period-luminosity relation makes longer-period Cepheids brighter
     (more negative M), and recovers a Cepheid's distance from period + apparent mag.
  4. Apparent and absolute magnitudes invert through the modulus.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from standard_candle import (distance_modulus, distance_from_modulus,  # noqa: E402
                             apparent_magnitude, absolute_magnitude, flux_ratio,
                             cepheid_absolute_magnitude, cepheid_distance)


def test_modulus_at_10pc():
    assert abs(distance_modulus(10.0)) < 1e-12


def test_lmc_modulus():
    assert abs(distance_modulus(50000.0) - 18.5) < 0.05


def test_modulus_inverts():
    for d in (10.0, 100.0, 5e4, 1e8):
        assert abs(distance_from_modulus(distance_modulus(d)) - d) / d < 1e-9


def test_five_magnitudes_is_100():
    assert abs(flux_ratio(5.0) - 0.01) < 1e-9
    assert abs(flux_ratio(-5.0) - 100.0) < 1e-6


def test_magnitude_inversion():
    M, d = -4.0, 2000.0
    m = apparent_magnitude(M, d)
    assert abs(absolute_magnitude(m, d) - M) < 1e-9


def test_cepheid_longer_brighter():
    assert cepheid_absolute_magnitude(30.0) < cepheid_absolute_magnitude(3.0)


def test_cepheid_distance_roundtrip():
    # place a P=10 d Cepheid at 1000 pc, compute its apparent mag, recover distance
    P, d = 10.0, 1000.0
    M = cepheid_absolute_magnitude(P)
    m = apparent_magnitude(M, d)
    assert abs(cepheid_distance(P, m) - d) < 1e-6


def test_farther_fainter():
    M = -4.0
    assert apparent_magnitude(M, 10000.0) > apparent_magnitude(M, 100.0)


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
