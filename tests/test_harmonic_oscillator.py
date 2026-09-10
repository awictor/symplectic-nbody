"""Quantum-harmonic-oscillator tests.

Claims checked:
  1. Energy levels are equally spaced by hbar omega (unlike the n^2 box).
  2. The ground state is nonzero: E_0 = (1/2) hbar omega (zero-point energy).
  3. The CO molecule (k~1900 N/m) vibrates with a ~0.27 eV quantum, absorbing near
     4.6 microns in the infrared.
  4. omega ~ sqrt(k/m); the spring constant inverts from the frequency.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harmonic_oscillator import (angular_frequency, energy_level,  # noqa: E402
                                 zero_point_energy, level_spacing,
                                 transition_wavelength, turning_point,
                                 spring_constant_from_frequency, EV, AMU)

MU_CO = 12 * 16 / 28.0 * AMU
K_CO = 1902.0


def test_equal_spacing():
    s1 = energy_level(1, K_CO, MU_CO) - energy_level(0, K_CO, MU_CO)
    s2 = energy_level(5, K_CO, MU_CO) - energy_level(4, K_CO, MU_CO)
    assert abs(s1 - s2) < 1e-30
    assert abs(s1 - level_spacing(K_CO, MU_CO)) < 1e-30


def test_zero_point_nonzero():
    e0 = zero_point_energy(K_CO, MU_CO)
    assert e0 > 0.0
    assert abs(e0 - 0.5 * level_spacing(K_CO, MU_CO)) < 1e-30


def test_co_vibrational_quantum():
    assert abs(level_spacing(K_CO, MU_CO) / EV - 0.27) < 0.01


def test_co_infrared_wavelength():
    lam = transition_wavelength(K_CO, MU_CO) * 1e6  # microns
    assert abs(lam - 4.6) < 0.2, f"CO vibration {lam} um not ~4.6"


def test_omega_sqrt_k_over_m():
    assert abs(angular_frequency(4.0, 1.0) / angular_frequency(1.0, 1.0) - 2.0) < 1e-9


def test_stiffer_higher_energy():
    assert energy_level(0, 2 * K_CO, MU_CO) > energy_level(0, K_CO, MU_CO)


def test_spring_constant_roundtrip():
    f = angular_frequency(K_CO, MU_CO) / (2 * math.pi)
    assert abs(spring_constant_from_frequency(f, MU_CO) - K_CO) < 1e-6


def test_turning_point_grows_with_n():
    assert turning_point(3, K_CO, MU_CO) > turning_point(0, K_CO, MU_CO)


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
