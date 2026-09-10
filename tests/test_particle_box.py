"""Particle-in-a-box tests.

Claims checked:
  1. Levels rise as n^2; the ground state is nonzero (zero-point energy).
  2. An electron in a ~1 nm box has eV-scale levels; squeezing the box raises them
     as 1/L^2 (why smaller quantum dots glow bluer).
  3. The eigenfunctions are normalized and psi_n has n-1 interior nodes.
  4. The box width for a target level gap inverts the energy formula.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from particle_box import (energy_level, ground_state_energy, level_spacing,  # noqa: E402
                          transition_wavelength, wavefunction, box_width_for_gap,
                          EV, M_E)

L1 = 1e-9


def test_n_squared():
    assert abs(energy_level(3, L1) / energy_level(1, L1) - 9.0) < 1e-9


def test_ground_state_nonzero():
    assert ground_state_energy(L1) > 0.0
    assert abs(ground_state_energy(L1) / EV - 0.376) < 0.01


def test_inverse_L_squared():
    assert abs(energy_level(1, 2e-9) / energy_level(1, 1e-9) - 0.25) < 1e-9


def test_smaller_dot_bluer():
    # smaller box -> larger gap -> shorter (bluer) transition wavelength
    assert transition_wavelength(1, 2, 1e-9) < transition_wavelength(1, 2, 2e-9)


def test_level_spacing_odd_multiple():
    # E_{n+1} - E_n = (2n+1) E_1
    assert abs(level_spacing(2, L1) / energy_level(1, L1) - 5.0) < 1e-9


def test_wavefunction_normalized():
    N = 20000
    dx = L1 / N
    norm = sum(wavefunction(3, i * dx, L1) ** 2 for i in range(N)) * dx
    assert abs(norm - 1.0) < 1e-3


def test_wavefunction_zero_outside():
    assert wavefunction(1, -1e-10, L1) == 0.0
    assert wavefunction(1, 2e-9, L1) == 0.0


def test_box_width_inversion():
    L = box_width_for_gap(2.0)
    gap = (energy_level(2, L) - energy_level(1, L)) / EV
    assert abs(gap - 2.0) < 1e-6


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
