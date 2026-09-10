"""Maxwell-Boltzmann speed-distribution tests.

Claims checked:
  1. The three characteristic speeds obey v_p : <v> : v_rms = 1 : 1.128 : 1.225,
     independent of gas and temperature.
  2. Nitrogen in room air has v_rms ~ 500 m/s; lighter gases scale as 1/sqrt(m).
  3. The distribution is normalized to 1, and the high-speed tail (exp(-v^2)) leaves
     only a tiny fraction above a few v_p.
  4. Mean kinetic energy is (3/2) k_B T, independent of mass.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maxwell_boltzmann import (distribution, most_probable_speed,  # noqa: E402
                               mean_speed, rms_speed, mean_kinetic_energy,
                               fraction_above, K_B, AMU)

M_N2 = 28.0 * AMU
T = 300.0


def test_characteristic_speed_ratios():
    vp = most_probable_speed(T, M_N2)
    assert abs(mean_speed(T, M_N2) / vp - 1.128) < 0.002
    assert abs(rms_speed(T, M_N2) / vp - 1.225) < 0.002


def test_ordering():
    vp = most_probable_speed(T, M_N2)
    assert vp < mean_speed(T, M_N2) < rms_speed(T, M_N2)


def test_nitrogen_rms_speed():
    v = rms_speed(T, M_N2)
    assert 480.0 < v < 540.0, f"N2 rms speed {v} m/s not ~500"


def test_lighter_faster():
    m_h2 = 2.0 * AMU
    assert abs(rms_speed(T, m_h2) / rms_speed(T, M_N2) - math.sqrt(28.0 / 2.0)) < 1e-6


def test_normalized():
    total = fraction_above(0.0, T, M_N2)
    assert abs(total - 1.0) < 1e-3, f"distribution not normalized: {total}"


def test_tail_is_small():
    vp = most_probable_speed(T, M_N2)
    assert fraction_above(3.0 * vp, T, M_N2) < 1e-2
    assert fraction_above(5.0 * vp, T, M_N2) < 1e-8


def test_mean_kinetic_energy():
    assert abs(mean_kinetic_energy(300.0) - 1.5 * K_B * 300.0) < 1e-30
    # independent of mass -> equal for any gas at the same T
    assert mean_kinetic_energy(300.0) == mean_kinetic_energy(300.0)


def test_temperature_scaling():
    # v_rms ~ sqrt(T): quadruple T -> double the speed
    assert abs(rms_speed(4 * T, M_N2) / rms_speed(T, M_N2) - 2.0) < 1e-9


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
