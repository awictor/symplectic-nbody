"""Joule-Thomson tests.

Claims checked:
  1. Nitrogen (inversion temperature well above room temperature) cools when
     throttled; hydrogen and helium (low inversion temperatures) warm unless
     pre-cooled.
  2. The maximum inversion temperature is T_inv = 2a/Rb = (27/4) T_c.
  3. Below the inversion temperature mu_JT > 0 (cooling); above it mu_JT < 0.
  4. An ideal gas (a=0, b=0) has mu_JT = 0 -- no Joule-Thomson effect.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from joule_thomson import (jt_coefficient, inversion_temperature,  # noqa: E402
                           cools_on_expansion, temperature_change,
                           inversion_over_critical, A_N2, B_N2, CP_N2,
                           A_H2, B_H2, CP_H2, A_HE, B_HE, CP_HE)


def test_nitrogen_cools_at_room_temp():
    assert cools_on_expansion(300.0, A_N2, B_N2)
    assert jt_coefficient(300.0, A_N2, B_N2, CP_N2) > 0.0


def test_hydrogen_helium_warm():
    assert not cools_on_expansion(300.0, A_H2, B_H2)
    assert not cools_on_expansion(300.0, A_HE, B_HE)


def test_inversion_over_critical():
    for a, b in ((A_N2, B_N2), (A_H2, B_H2), (A_HE, B_HE)):
        assert abs(inversion_over_critical(a, b) - 27.0 / 4.0) < 1e-9


def test_mu_sign_flips_at_inversion():
    T_inv = inversion_temperature(A_N2, B_N2)
    assert jt_coefficient(0.5 * T_inv, A_N2, B_N2, CP_N2) > 0.0
    assert jt_coefficient(1.5 * T_inv, A_N2, B_N2, CP_N2) < 0.0


def test_ideal_gas_no_effect():
    assert abs(jt_coefficient(300.0, 0.0, 0.0, CP_N2)) < 1e-30


def test_throttling_cools_nitrogen():
    # a pressure drop cools nitrogen (dT < 0)
    assert temperature_change(300.0, 10e6, A_N2, B_N2, CP_N2) < 0.0


def test_helium_lowest_inversion():
    assert (inversion_temperature(A_HE, B_HE)
            < inversion_temperature(A_H2, B_H2)
            < inversion_temperature(A_N2, B_N2))


def test_colder_stronger_cooling():
    # deeper below inversion -> larger positive mu (stronger cooling)
    assert (jt_coefficient(150.0, A_N2, B_N2, CP_N2)
            > jt_coefficient(300.0, A_N2, B_N2, CP_N2))


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
