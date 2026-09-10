"""Sunyaev-Zeldovich-effect tests.

Claims checked:
  1. A massive cluster's Compton y-parameter is ~1e-5 to 1e-4.
  2. The y-parameter is linear in electron density, temperature, and path length.
  3. In the Rayleigh-Jeans limit the CMB shows a DECREMENT of -2 y T_CMB, a few
     hundred microkelvin for a massive cluster.
  4. The SZ signal is redshift-independent (a fractional distortion of the CMB).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sz import (y_parameter, y_from_kev, rj_temperature_decrement,  # noqa: E402
                rj_decrement_kelvin, is_redshift_independent, MPC, T_CMB)

N_E = 1e3        # /m^3
KT = 8.0         # keV
L = 1 * MPC


def test_y_parameter_scale():
    y = y_from_kev(N_E, KT, L)
    assert 1e-6 < y < 1e-3, f"cluster y {y} off scale"


def test_y_linear_in_density():
    assert abs(y_from_kev(2 * N_E, KT, L) / y_from_kev(N_E, KT, L) - 2.0) < 1e-9


def test_y_linear_in_temperature():
    assert abs(y_from_kev(N_E, 2 * KT, L) / y_from_kev(N_E, KT, L) - 2.0) < 1e-9


def test_y_linear_in_path():
    assert abs(y_from_kev(N_E, KT, 2 * L) / y_from_kev(N_E, KT, L) - 2.0) < 1e-9


def test_rj_decrement_negative():
    y = y_from_kev(N_E, KT, L)
    assert rj_temperature_decrement(y) < 0, "RJ SZ should be a decrement"
    assert abs(rj_temperature_decrement(y) - (-2.0 * y)) < 1e-15


def test_decrement_microkelvin_scale():
    dT_uK = abs(rj_decrement_kelvin(y_from_kev(N_E, KT, L))) * 1e6
    assert 50 < dT_uK < 1000, f"decrement {dT_uK} uK off scale"


def test_redshift_independent():
    assert is_redshift_independent()


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
