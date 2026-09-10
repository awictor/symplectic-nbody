"""Adiabatic-process tests.

Claims checked:
  1. The adiabatic (Laplace) speed of sound in air at 293 K is ~343 m/s, ~18% above
     Newton's isothermal estimate.
  2. Diesel-ratio compression (22:1) heats air past autoignition (~1000 K); expansion
     cools.
  3. P V^gamma and T V^(gamma-1) are invariant along an adiabat.
  4. Adiabatic work is (P1 V1 - P2 V2)/(gamma-1), positive for expansion.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adiabatic import (pressure_after, temperature_after_volume,  # noqa: E402
                       temperature_after_pressure, adiabatic_work, sound_speed,
                       compression_ratio_for_temperature, R_GAS,
                       GAMMA_DIATOMIC, GAMMA_MONO)

M_AIR = 0.02896


def test_sound_speed_air():
    c = sound_speed(293.0, M_AIR)
    assert abs(c - 343.0) < 3.0, f"speed of sound {c} m/s not ~343"


def test_laplace_beats_newton():
    adiabatic = sound_speed(293.0, M_AIR, GAMMA_DIATOMIC)
    isothermal = sound_speed(293.0, M_AIR, 1.0)
    assert abs(adiabatic / isothermal - math.sqrt(GAMMA_DIATOMIC)) < 1e-9


def test_diesel_compression_ignites():
    T2 = temperature_after_volume(300.0, 22.0, 1.0)
    assert T2 > 800.0, f"diesel compression {T2} K should exceed autoignition"


def test_expansion_cools():
    assert temperature_after_volume(300.0, 1.0, 10.0) < 300.0


def test_pvgamma_invariant():
    P1, V1 = 1e5, 1.0
    V2 = 0.5
    P2 = pressure_after(P1, V1, V2)
    assert abs(P1 * V1 ** GAMMA_DIATOMIC - P2 * V2 ** GAMMA_DIATOMIC) < 1.0


def test_tv_invariant():
    T1, V1, V2 = 300.0, 1.0, 0.5
    T2 = temperature_after_volume(T1, V1, V2)
    assert abs(T1 * V1 ** (GAMMA_DIATOMIC - 1) - T2 * V2 ** (GAMMA_DIATOMIC - 1)) < 1e-9


def test_pressure_temperature_consistency():
    # cross-check: temperature via P should match temperature via V
    P1, V1, V2 = 1e5, 1.0, 0.5
    P2 = pressure_after(P1, V1, V2)
    Tv = temperature_after_volume(300.0, V1, V2)
    Tp = temperature_after_pressure(300.0, P1, P2)
    assert abs(Tv - Tp) < 1e-6


def test_expansion_work_positive():
    W = adiabatic_work(1e5, 1.0, pressure_after(1e5, 1.0, 2.0), 2.0)
    assert W > 0.0   # gas expands, does work


def test_compression_ratio_inverts():
    T2 = temperature_after_volume(300.0, 22.0, 1.0)
    r = compression_ratio_for_temperature(300.0, T2)
    assert abs(r - 22.0) < 1e-6


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
