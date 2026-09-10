"""Galaxy-cluster virial-temperature tests.

Claims checked:
  1. A Coma-like cluster (1e15 M_sun, 2 Mpc) has a virial temperature of a few
     keV (~10^8 K) -- hot enough to emit X-rays.
  2. The temperature from a ~1000 km/s velocity dispersion agrees with the
     mass/radius estimate.
  3. The mass-temperature relation kT ~ M^{2/3} holds at fixed overdensity.
  4. The mass-from-temperature inversion recovers the input mass.
  5. Bremsstrahlung emissivity rises with density and temperature.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cluster import (virial_temperature, temperature_from_dispersion,  # noqa: E402
                     kT_kev, mass_from_temperature, bremsstrahlung_scaling,
                     M_SUN, MPC, K_B, KEV)


def test_coma_temperature():
    kt = kT_kev(1e15 * M_SUN, 2 * MPC)
    assert 4.0 < kt < 12.0, f"Coma kT {kt} keV not ~8"
    T = virial_temperature(1e15 * M_SUN, 2 * MPC)
    assert 3e7 < T < 2e8, f"Coma T {T} K not ~1e8"


def test_dispersion_matches_mass_estimate():
    # sigma ~ 1000 km/s should give a similar keV to the M/R estimate
    kt_sigma = temperature_from_dispersion(1e6) * K_B / KEV
    assert 4.0 < kt_sigma < 10.0, f"dispersion kT {kt_sigma} keV off"


def test_mass_temperature_relation():
    # at fixed overdensity R ~ M^{1/3}, so 8x mass -> 2x radius -> kT x4
    base = kT_kev(1e15 * M_SUN, 2 * MPC)
    bigger = kT_kev(8e15 * M_SUN, 4 * MPC)
    assert abs(bigger / base - 4.0) < 1e-6, "kT should scale as M^{2/3}"


def test_mass_inversion():
    R = 2 * MPC
    kt = kT_kev(1e15 * M_SUN, R)
    M_rec = mass_from_temperature(kt, R)
    assert abs(M_rec - 1e15 * M_SUN) / (1e15 * M_SUN) < 1e-6


def test_bremsstrahlung_scaling():
    assert bremsstrahlung_scaling(2.0, 1.0) > bremsstrahlung_scaling(1.0, 1.0)
    assert bremsstrahlung_scaling(1.0, 4.0) > bremsstrahlung_scaling(1.0, 1.0)


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
