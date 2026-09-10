"""Thermal bremsstrahlung (free-free) tests.

Claims checked:
  1. Emissivity scales as n^2 (two-body) and sqrt(T).
  2. Cooling time scales as sqrt(T)/n, so dense cluster cores cool within a
     Hubble time (cooling flows) while tenuous outskirts effectively never cool.
  3. The cluster-gas regime (~10^7-10^8 K, ~1e2-1e4 /m^3) gives Gyr-scale to
     never cooling times.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bremsstrahlung import (emissivity, cooling_time, cooling_time_gyr,  # noqa: E402
                            cools_within_hubble)


def test_emissivity_density_squared():
    assert abs(emissivity(2e4, 5e7) / emissivity(1e4, 5e7) - 4.0) < 1e-9


def test_emissivity_sqrt_temperature():
    assert abs(emissivity(1e4, 4 * 5e7) / emissivity(1e4, 5e7) - 2.0) < 1e-9


def test_cooling_time_scalings():
    base = cooling_time(1e4, 5e7)
    assert abs(cooling_time(2e4, 5e7) / base - 0.5) < 1e-9, "t ~ 1/n"
    assert abs(cooling_time(1e4, 4 * 5e7) / base - 2.0) < 1e-9, "t ~ sqrt(T)"


def test_dense_core_cools():
    assert cools_within_hubble(1e4, 5e7), "dense cluster core should cool (flow)"
    assert cooling_time_gyr(1e4, 5e7) < 13.8


def test_outskirts_never_cool():
    assert not cools_within_hubble(100.0, 1e8), "tenuous outskirts should not cool"
    assert cooling_time_gyr(100.0, 1e8) > 100.0


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
