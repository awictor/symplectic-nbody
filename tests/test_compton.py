"""Compton / inverse-Compton scattering tests.

Claims checked:
  1. The Compton wavelength is 2.426 pm and the electron rest energy is 511 keV.
  2. The Compton shift is lambda_C at 90 deg, 2 lambda_C at 180 deg, and zero at
     forward scattering.
  3. A photon LOSES energy scattering off a stationary electron, most at
     back-scattering.
  4. Inverse Compton boosts a photon by ~(4/3) gamma^2, turning a CMB photon into
     an X-ray for gamma ~ 1000.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compton import (compton_shift, scattered_wavelength, scattered_energy,  # noqa: E402
                     inverse_compton_boost, electron_rest_energy_kev,
                     LAMBDA_C, KEV)


def test_compton_wavelength():
    assert abs(LAMBDA_C * 1e12 - 2.426) < 0.001


def test_electron_rest_energy():
    assert abs(electron_rest_energy_kev() - 511.0) < 1.0


def test_shift_angles():
    assert abs(compton_shift(0.0)) < 1e-30, "no shift forward"
    assert abs(compton_shift(math.pi / 2) - LAMBDA_C) < 1e-18, "shift = lambda_C at 90 deg"
    assert abs(compton_shift(math.pi) - 2 * LAMBDA_C) < 1e-18, "shift = 2 lambda_C at 180 deg"


def test_photon_loses_energy():
    E0 = 100 * KEV
    assert scattered_energy(E0, math.pi) < E0, "photon should lose energy"
    # back-scatter loses the most
    assert scattered_energy(E0, math.pi) < scattered_energy(E0, math.pi / 2)


def test_scattered_wavelength_increases():
    lam0 = 1e-11
    assert scattered_wavelength(lam0, math.pi) > lam0


def test_inverse_compton_boost():
    b = inverse_compton_boost(100.0)
    assert abs(b - (4.0 / 3.0) * 100.0 ** 2) / b < 0.01, "IC boost ~ 4/3 gamma^2"
    # CMB photon (1 meV) -> ~keV X-ray at gamma=1000
    boosted_ev = 1e-3 * inverse_compton_boost(1000.0)
    assert 500 < boosted_ev < 3000, f"boosted CMB photon {boosted_ev} eV not ~keV"


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
