"""Blackbody-radiation tests.

Claims checked:
  1. Wien's law: the Sun (~5772 K) peaks near 500 nm, the CMB (2.725 K) near
     1 mm, the human body (310 K) near 10 microns.
  2. The numeric Planck peak matches Wien's displacement law.
  3. Stefan-Boltzmann: total flux scales as T^4, and the Sun's luminosity is
     ~3.83e26 W.
  4. lambda_max T is a constant (Wien) independent of temperature.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blackbody import (planck_wavelength, wien_peak_wavelength,  # noqa: E402
                       stefan_boltzmann_flux, luminosity,
                       peak_wavelength_numeric, WIEN_B)


def test_sun_peak_wavelength():
    lam = wien_peak_wavelength(5772) * 1e9
    assert 480 < lam < 520, f"Sun peak {lam} nm not ~500"


def test_cmb_peak_wavelength():
    lam = wien_peak_wavelength(2.725) * 1e3
    assert 1.0 < lam < 1.1, f"CMB peak {lam} mm not ~1.06"


def test_body_peak_wavelength():
    lam = wien_peak_wavelength(310) * 1e6
    assert 8 < lam < 11, f"body peak {lam} um not ~9.3"


def test_numeric_peak_matches_wien():
    for T in (300.0, 5772.0):
        num = peak_wavelength_numeric(T)
        wien = wien_peak_wavelength(T)
        assert abs(num - wien) / wien < 0.02, f"numeric peak {num} vs Wien {wien} at {T} K"


def test_stefan_boltzmann_t4():
    r = stefan_boltzmann_flux(2000.0) / stefan_boltzmann_flux(1000.0)
    assert abs(r - 16.0) < 1e-9, "flux should scale as T^4"


def test_solar_luminosity():
    L = luminosity(6.957e8, 5772.0)
    assert abs(L - 3.83e26) / 3.83e26 < 0.02, f"solar luminosity {L} not ~3.83e26 W"


def test_wien_constant():
    # lambda_max * T is constant across temperatures
    for T in (100.0, 1000.0, 10000.0):
        assert abs(wien_peak_wavelength(T) * T - WIEN_B) < 1e-9


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
