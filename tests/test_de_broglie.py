"""de Broglie matter-wave tests.

Claims checked:
  1. A 100 keV electron has lambda ~ 4 pm (why electron microscopes beat light).
  2. A room-temperature thermal neutron has lambda ~ 0.1 nm -- atomic spacing, ideal
     for neutron diffraction.
  3. Everyday objects have utterly negligible wavelengths (a baseball ~ 10^-34 m).
  4. lambda ~ 1/p and ~ 1/sqrt(E); momentum and wavelength invert.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from de_broglie import (wavelength_from_momentum, wavelength_from_energy,  # noqa: E402
                        wavelength_from_velocity, thermal_wavelength,
                        momentum_from_wavelength, electron_microscope_wavelength,
                        M_E, M_N, EV, H)


def test_electron_microscope():
    lam = electron_microscope_wavelength(1e5) * 1e12  # pm
    assert 3.0 < lam < 5.0, f"100 keV electron wavelength {lam} pm off scale"


def test_thermal_neutron():
    lam = thermal_wavelength(300.0, M_N) * 1e9  # nm
    assert 0.05 < lam < 0.3, f"thermal neutron wavelength {lam} nm off scale"


def test_baseball_negligible():
    lam = wavelength_from_velocity(40.0, 0.145)
    assert lam < 1e-30


def test_inverse_momentum():
    assert abs(wavelength_from_momentum(2e-24) / wavelength_from_momentum(4e-24) - 2.0) < 1e-9


def test_inverse_sqrt_energy():
    l1 = wavelength_from_energy(1.0 * EV, M_E)
    l4 = wavelength_from_energy(4.0 * EV, M_E)
    assert abs(l1 / l4 - 2.0) < 1e-9


def test_momentum_wavelength_invert():
    for lam in (1e-10, 1e-11, 1e-12):
        assert abs(wavelength_from_momentum(momentum_from_wavelength(lam)) - lam) < 1e-20


def test_energy_form_matches_momentum_form():
    # lambda from E should equal lambda from p = sqrt(2 m E)
    E = 100.0 * EV
    p = math.sqrt(2.0 * M_E * E)
    assert abs(wavelength_from_energy(E, M_E) - wavelength_from_momentum(p)) < 1e-20


def test_heavier_shorter():
    # at the same kinetic energy, a heavier particle has a shorter wavelength
    assert wavelength_from_energy(1.0 * EV, M_N) < wavelength_from_energy(1.0 * EV, M_E)


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
