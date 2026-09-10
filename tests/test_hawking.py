"""Hawking / black-hole-thermodynamics tests.

Claims checked:
  1. The Hawking temperature is inversely proportional to mass, and a solar-mass
     hole sits at ~62 nanokelvin.
  2. The evaporation time scales as M^3, and a solar-mass hole lives ~2e67 years
     (vastly longer than the age of the universe).
  3. A primordial black hole evaporating within a Hubble time has a mass of
     ~1.7e11 kg -- and the mass/lifetime inversion is self-consistent.
  4. The Bekenstein-Hawking entropy is one quarter of the horizon area in Planck
     units, S/k_B = 4 pi G M^2 / (hbar c), and is astronomically large.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hawking import (hawking_temperature, evaporation_time, mass_evaporating_in,  # noqa: E402
                     entropy_over_kb, horizon_area, bekenstein_hawking_entropy,
                     M_SUN, YEAR, L_P2, K_B)


def test_temperature_inverse_mass():
    assert abs(hawking_temperature(2 * M_SUN) / hawking_temperature(M_SUN) - 0.5) < 1e-12


def test_solar_temperature_is_nanokelvin():
    T = hawking_temperature(M_SUN)
    assert 5e-8 < T < 7e-8, f"solar-mass Hawking T {T} not ~62 nK"


def test_evaporation_scales_as_mass_cubed():
    ratio = evaporation_time(2 * M_SUN) / evaporation_time(M_SUN)
    assert abs(ratio - 8.0) < 1e-9, f"evaporation should scale as M^3, ratio {ratio}"


def test_solar_evaporation_far_exceeds_universe_age():
    t_yr = evaporation_time(M_SUN) / YEAR
    assert t_yr > 1e60, f"solar-mass evaporation {t_yr} yr should dwarf the universe age"


def test_primordial_mass_evaporating_now():
    t_universe = 13.8e9 * YEAR
    M = mass_evaporating_in(t_universe)
    assert 1e11 < M < 3e11, f"primordial mass {M} kg not ~1.7e11"
    # inversion is self-consistent
    assert abs(evaporation_time(M) - t_universe) / t_universe < 1e-9


def test_entropy_is_quarter_area():
    # S/k_B should equal A / (4 l_p^2)
    M = 10 * M_SUN
    assert abs(entropy_over_kb(M) - horizon_area(M) / (4.0 * L_P2)) / entropy_over_kb(M) < 1e-9
    # and be enormous for a stellar hole
    assert entropy_over_kb(M_SUN) > 1e76


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
