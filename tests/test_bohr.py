"""Bohr-model / Rydberg tests.

Claims checked:
  1. The ground-state energy is -13.6 eV (the ionization energy) and the Bohr radius
     is ~52.9 pm.
  2. Balmer H-alpha (n=3->2) is 656.3 nm (visible red); Lyman-alpha (2->1) is 121.6 nm
     (ultraviolet).
  3. Energy goes as -1/n^2 and radius as n^2.
  4. The n=1 orbital speed is alpha*c, giving v/c ~ 1/137 (the fine-structure constant).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bohr import (energy_level_ev, radius, transition_wavelength,  # noqa: E402
                  transition_energy_ev, ionization_energy_ev, orbital_speed,
                  BOHR_RADIUS, C)


def test_ground_energy():
    assert abs(energy_level_ev(1) + 13.6) < 0.02


def test_bohr_radius():
    assert abs(BOHR_RADIUS * 1e12 - 52.92) < 0.05


def test_h_alpha():
    lam = transition_wavelength(2, 3) * 1e9
    assert abs(lam - 656.3) < 0.5, f"H-alpha {lam} nm not ~656.3"


def test_lyman_alpha():
    lam = transition_wavelength(1, 2) * 1e9
    assert abs(lam - 121.6) < 0.5, f"Lyman-alpha {lam} nm not ~121.6"


def test_energy_inverse_n_squared():
    assert abs(energy_level_ev(2) / energy_level_ev(1) - 0.25) < 1e-9


def test_radius_n_squared():
    assert abs(radius(3) / radius(1) - 9.0) < 1e-9


def test_ionization_energy():
    assert abs(ionization_energy_ev(1) - 13.6) < 0.02
    assert abs(ionization_energy_ev(2) - 3.4) < 0.02


def test_fine_structure_constant():
    alpha = orbital_speed(1) / C
    assert abs(1.0 / alpha - 137.036) < 0.5


def test_transition_energy_matches_wavelength():
    # E = hc/lambda should match the level-difference energy
    lam = transition_wavelength(2, 3)
    E_photon = 6.62607015e-34 * C / lam / 1.602176634e-19  # eV
    assert abs(E_photon - transition_energy_ev(2, 3)) < 1e-3


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
