"""Olbers'-paradox tests.

Claims checked:
  1. The mean free path to a star (lambda = 1/n sigma) is astronomically large --
     ~10^16 light-years for realistic galaxy-star densities.
  2. The covering fraction is 1 - exp(-d/lambda): ~63% at one mean free path, ->1
     far beyond it (the sky would blaze).
  3. The horizon (c * age) is ~1.4x10^10 ly, vastly smaller than lambda, so the
     fraction of sky covered within the observable universe is tiny -- night is dark.
  4. An infinite static universe always blazes for any positive star density.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from olbers import (mean_free_path_to_star, covering_fraction,  # noqa: E402
                    horizon_distance, sky_fraction_within_horizon,
                    would_blaze_if_static, LY, PC, R_SUN)

N_STARS = 0.1 / PC ** 3   # ~0.1 stars per cubic parsec


def test_mean_free_path_huge():
    mfp_ly = mean_free_path_to_star(N_STARS) / LY
    assert mfp_ly > 1e14, f"mean free path {mfp_ly} ly not astronomically large"


def test_covering_one_mfp():
    mfp = mean_free_path_to_star(N_STARS)
    assert abs(covering_fraction(mfp, mfp) - (1.0 - 1.0 / math.e)) < 1e-9


def test_covering_saturates():
    mfp = mean_free_path_to_star(N_STARS)
    assert covering_fraction(10 * mfp, mfp) > 0.999


def test_horizon_scale():
    h_ly = horizon_distance(1.38e10) / LY
    assert abs(h_ly - 1.38e10) < 1e6


def test_night_is_dark():
    frac = sky_fraction_within_horizon(N_STARS)
    assert frac < 1e-5, f"sky fraction {frac} within horizon should be tiny"


def test_mfp_shorter_at_higher_density():
    assert mean_free_path_to_star(2 * N_STARS) < mean_free_path_to_star(N_STARS)


def test_static_universe_blazes():
    assert would_blaze_if_static(N_STARS)
    assert not would_blaze_if_static(0.0)


def test_covering_monotonic():
    mfp = mean_free_path_to_star(N_STARS)
    assert covering_fraction(2 * mfp, mfp) > covering_fraction(mfp, mfp)


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
