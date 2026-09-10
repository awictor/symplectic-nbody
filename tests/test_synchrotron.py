"""Synchrotron-radiation tests.

Claims checked:
  1. Relativistic electrons (gamma ~ 1e4) in microgauss fields radiate at radio
     frequencies (~GHz).
  2. The critical frequency scales as gamma^2 and as B.
  3. The single-electron power scales as gamma^2 and B^2, and higher-energy
     electrons cool faster (t ~ 1/(gamma B^2)).
  4. A power-law electron distribution N(E) ~ E^{-p} gives spectral index
     alpha = (p-1)/2 -- 0.75 for the canonical p = 2.5.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synchrotron import (gyrofrequency, critical_frequency,  # noqa: E402
                         single_electron_power, cooling_time, spectral_index)


def test_radio_frequency():
    nu = critical_frequency(1e4, 1e-9)
    assert 1e8 < nu < 1e11, f"critical freq {nu} Hz not radio"


def test_critical_frequency_scalings():
    assert abs(critical_frequency(2e4, 1e-9) / critical_frequency(1e4, 1e-9) - 4.0) < 1e-6
    assert abs(critical_frequency(1e4, 2e-9) / critical_frequency(1e4, 1e-9) - 2.0) < 1e-6


def test_power_scalings():
    base = single_electron_power(1e4, 1e-9)
    assert abs(single_electron_power(1e4, 2e-9) / base - 4.0) < 1e-6, "P ~ B^2"
    assert abs(single_electron_power(2e4, 1e-9) / base - 4.0) < 1e-3, "P ~ gamma^2"


def test_higher_energy_cools_faster():
    assert cooling_time(2e4, 1e-9) < cooling_time(1e4, 1e-9)
    # stronger field cools faster too
    assert cooling_time(1e4, 2e-9) < cooling_time(1e4, 1e-9)


def test_spectral_index():
    assert abs(spectral_index(2.5) - 0.75) < 1e-9
    assert abs(spectral_index(2.0) - 0.5) < 1e-9
    # steeper electron spectrum -> steeper radio spectrum
    assert spectral_index(3.0) > spectral_index(2.0)


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
