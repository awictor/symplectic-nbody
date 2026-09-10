"""Tolman surface-brightness-dimming tests.

Claims checked:
  1. Expanding-universe dimming is (1+z)^-4: 1/16 at z=1, 1/256 at z=3.
  2. In magnitudes that is 2.5*4*log10(1+z) ~ 3.01 mag at z=1.
  3. A static 'tired-light' universe dims only as (1+z)^-1, so the expanding
     prediction is (1+z)^-3 fainter.
  4. The exponent recovered from an observed ratio is ~4 for expansion; there is no
     dimming at z=0.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tolman import (dimming_factor, dimming_magnitudes, tired_light_factor,  # noqa: E402
                    expanding_vs_tired, exponent_from_observation)


def test_z1_is_one_sixteenth():
    assert abs(dimming_factor(1.0) - 1.0 / 16.0) < 1e-12


def test_z3_is_one_over_256():
    assert abs(dimming_factor(3.0) - 1.0 / 256.0) < 1e-12


def test_no_dimming_at_z0():
    assert abs(dimming_factor(0.0) - 1.0) < 1e-12


def test_magnitudes_at_z1():
    assert abs(dimming_magnitudes(1.0) - 3.01) < 0.02


def test_tired_light_single_factor():
    assert abs(tired_light_factor(1.0) - 0.5) < 1e-12
    assert abs(tired_light_factor(3.0) - 0.25) < 1e-12


def test_expanding_fainter_than_tired():
    # expanding/tired = (1+z)^-3
    assert abs(expanding_vs_tired(1.0) - 1.0 / 8.0) < 1e-12
    assert expanding_vs_tired(2.0) < expanding_vs_tired(1.0)


def test_exponent_recovery():
    assert abs(exponent_from_observation(1.0, dimming_factor(1.0)) - 4.0) < 1e-9
    assert abs(exponent_from_observation(2.0, tired_light_factor(2.0)) - 1.0) < 1e-9


def test_dimming_monotonic():
    assert dimming_factor(0.5) > dimming_factor(1.0) > dimming_factor(3.0)


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
