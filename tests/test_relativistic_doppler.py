"""Relativistic-Doppler tests.

Claims checked:
  1. Radial motion: receding redshifts (z>0), approaching blueshifts (z<0), and the
     two frequency ratios are reciprocals.
  2. The transverse Doppler shift is a pure time-dilation redshift 1/gamma, present
     even with no line-of-sight motion (the Ives-Stilwell effect).
  3. The general-angle factor reduces to the radial and transverse cases at
     theta = 0, pi/2, pi.
  4. Redshift and velocity invert: beta = ((1+z)^2-1)/((1+z)^2+1).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativistic_doppler import (lorentz_factor, frequency_ratio_radial,  # noqa: E402
                                  wavelength_ratio_radial, redshift_radial,
                                  transverse_frequency_ratio, transverse_redshift,
                                  frequency_ratio_angle, beta_from_redshift)


def test_receding_redshift():
    assert redshift_radial(0.5) > 0.0
    assert abs(redshift_radial(0.5) - 0.7321) < 1e-3


def test_approaching_blueshift():
    assert redshift_radial(0.5, approaching=True) < 0.0


def test_frequency_wavelength_reciprocal():
    assert abs(frequency_ratio_radial(0.5) * wavelength_ratio_radial(0.5) - 1.0) < 1e-12


def test_transverse_is_time_dilation():
    beta = 0.5
    assert abs(transverse_frequency_ratio(beta) - 1.0 / lorentz_factor(beta)) < 1e-12
    assert transverse_redshift(beta) > 0.0   # always a redshift


def test_angle_reduces_to_transverse():
    beta = 0.6
    assert abs(frequency_ratio_angle(beta, math.pi / 2.0)
               - transverse_frequency_ratio(beta)) < 1e-12


def test_angle_reduces_to_radial():
    beta = 0.6
    # theta = pi (source moving away along the line of sight) -> receding radial
    assert abs(frequency_ratio_angle(beta, math.pi)
               - frequency_ratio_radial(beta)) < 1e-9
    # theta = 0 (approaching) -> approaching radial
    assert abs(frequency_ratio_angle(beta, 0.0)
               - frequency_ratio_radial(beta, approaching=True)) < 1e-9


def test_beta_from_redshift():
    assert abs(beta_from_redshift(1.0) - 0.6) < 1e-9


def test_redshift_velocity_roundtrip():
    for beta in (0.1, 0.5, 0.9):
        z = redshift_radial(beta)
        assert abs(beta_from_redshift(z) - beta) < 1e-9


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
