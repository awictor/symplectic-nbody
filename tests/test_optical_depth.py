"""Optical-depth / radiative-transfer tests.

Claims checked:
  1. Transmitted intensity is exp(-tau): 1/e at tau=1, ~half at the photospheric
     tau = 2/3, negligible when tau >> 1.
  2. Optical depth is linear in density, cross-section, and path length.
  3. The mean free path is 1/(n sigma); the photosphere sits at the depth where
     the inward optical depth reaches 2/3.
  4. The thin/thick boundary is at tau = 1.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optical_depth import (optical_depth, transmitted_fraction,  # noqa: E402
                           mean_free_path, is_optically_thick,
                           photosphere_depth, PHOTOSPHERE_TAU)

SIGMA = 6.6525e-29
N = 1e20
L = 1e6


def test_transmission_law():
    assert abs(transmitted_fraction(1.0) - 1.0 / math.e) < 1e-9
    assert abs(transmitted_fraction(PHOTOSPHERE_TAU) - 0.5) < 0.02
    assert transmitted_fraction(10.0) < 1e-3


def test_optical_depth_linear():
    base = optical_depth(N, SIGMA, L)
    assert abs(optical_depth(2 * N, SIGMA, L) / base - 2.0) < 1e-9
    assert abs(optical_depth(N, SIGMA, 2 * L) / base - 2.0) < 1e-9


def test_mean_free_path():
    assert abs(mean_free_path(N, SIGMA) - 1.0 / (N * SIGMA)) < 1e-6 * mean_free_path(N, SIGMA)
    # denser gas -> shorter mean free path
    assert mean_free_path(2 * N, SIGMA) < mean_free_path(N, SIGMA)


def test_thin_thick_boundary():
    assert not is_optically_thick(0.5)
    assert is_optically_thick(2.0)


def test_photosphere_depth():
    d = photosphere_depth(N, SIGMA)
    # optical depth measured over that depth should be exactly 2/3
    assert abs(optical_depth(N, SIGMA, d) - PHOTOSPHERE_TAU) < 1e-9


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
