"""Photoelectric-effect tests.

Claims checked:
  1. Sodium (phi=2.28 eV) has a threshold wavelength ~544 nm; longer (redder) light
     ejects nothing however intense.
  2. K_max = hf - phi rises linearly with frequency; the stopping voltage equals
     K_max in eV.
  3. Below threshold no electrons are emitted (K_max clamped to 0).
  4. The slope of K_max vs frequency is Planck's constant, independent of the metal.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photoelectric import (photon_energy, photon_energy_from_wavelength,  # noqa: E402
                           max_kinetic_energy, threshold_frequency,
                           threshold_wavelength, stopping_voltage, is_emitting,
                           PHI_SODIUM, PHI_CESIUM, PHI_ZINC, H, C, EV)


def test_sodium_threshold_wavelength():
    lam = threshold_wavelength(PHI_SODIUM) * 1e9
    assert abs(lam - 544.0) < 3.0, f"Na threshold {lam} nm not ~544"


def test_red_light_no_emission():
    f = C / 600e-9   # 600 nm, below Na threshold
    assert not is_emitting(f, PHI_SODIUM)
    assert max_kinetic_energy(f, PHI_SODIUM) == 0.0


def test_violet_ejects():
    f = C / 400e-9
    assert is_emitting(f, PHI_SODIUM)
    assert abs(max_kinetic_energy(f, PHI_SODIUM) / EV - 0.82) < 0.02


def test_stopping_voltage_equals_kmax_ev():
    f = C / 400e-9
    V = stopping_voltage(f, PHI_SODIUM)
    assert abs(V - max_kinetic_energy(f, PHI_SODIUM) / EV) < 1e-9


def test_slope_is_planck():
    # d K_max / d f = h, independent of the metal
    f1, f2 = 2e15, 3e15   # above both Na and Zn thresholds
    slope_na = (max_kinetic_energy(f2, PHI_SODIUM) - max_kinetic_energy(f1, PHI_SODIUM)) / (f2 - f1)
    slope_zn = (max_kinetic_energy(f2, PHI_ZINC) - max_kinetic_energy(f1, PHI_ZINC)) / (f2 - f1)
    assert abs(slope_na - H) < 1e-40
    assert abs(slope_na - slope_zn) < 1e-40


def test_higher_work_function_higher_threshold():
    assert threshold_frequency(PHI_ZINC) > threshold_frequency(PHI_SODIUM)


def test_photon_energy_forms_agree():
    lam = 500e-9
    assert abs(photon_energy(C / lam) - photon_energy_from_wavelength(lam)) < 1e-30


def test_cesium_lowest_threshold():
    # cesium has the lowest work function -> longest threshold wavelength
    assert threshold_wavelength(PHI_CESIUM) > threshold_wavelength(PHI_SODIUM)


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
