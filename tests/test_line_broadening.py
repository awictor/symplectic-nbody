"""Spectral-line-broadening tests.

Claims checked:
  1. The thermal Doppler width of H-alpha at 6000 K is ~20 pm (~15 GHz), and the
     H thermal speed there is ~10 km/s.
  2. Doppler width scales as sqrt(T) and as 1/sqrt(m) -- hotter and lighter atoms
     give broader lines.
  3. In hot thin gas Doppler dominates; in a dense stellar photosphere pressure
     (collisional) broadening takes over.
  4. Natural width is A/(4 pi); Lorentzian widths add linearly.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from line_broadening import (thermal_speed, doppler_width,  # noqa: E402
                             doppler_width_wavelength, natural_width,
                             pressure_width, dominant_mechanism,
                             lorentzian_fwhm, AMU, C)

LAM = 656.3e-9
NU0 = C / LAM
M_H = 1.008 * AMU


def test_halpha_doppler_width():
    dl = doppler_width_wavelength(LAM, 6000.0, M_H) * 1e12  # pm
    assert 15.0 < dl < 30.0, f"H-alpha Doppler width {dl} pm off scale"


def test_thermal_speed():
    v = thermal_speed(6000.0, M_H) / 1e3
    assert abs(v - 9.9) < 1.0, f"H thermal speed {v} km/s not ~10"


def test_doppler_sqrt_T():
    base = doppler_width(NU0, 3000.0, M_H)
    assert abs(doppler_width(NU0, 12000.0, M_H) / base - 2.0) < 1e-9


def test_doppler_lighter_broader():
    light = doppler_width(NU0, 6000.0, M_H)
    heavy = doppler_width(NU0, 6000.0, 16.0 * AMU)  # oxygen
    assert light > heavy
    assert abs(light / heavy - math.sqrt(16.0 / 1.008)) < 1e-6  # ~sqrt(m ratio)


def test_natural_width_formula():
    assert abs(natural_width(6.5e7) - 6.5e7 / (4.0 * math.pi)) < 1.0


def test_dominant_switches_with_density():
    assert dominant_mechanism(NU0, 6000.0, M_H, 6.5e7, 1e6) == "Doppler"
    assert dominant_mechanism(NU0, 6000.0, M_H, 6.5e7, 1e11) == "pressure"


def test_lorentzian_widths_add():
    nat = natural_width(6.5e7)
    pre = pressure_width(1e9)
    assert abs(lorentzian_fwhm(nat, pre) - 2.0 * (nat + pre)) < 1e-6


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
