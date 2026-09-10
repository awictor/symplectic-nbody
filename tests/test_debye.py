"""Debye-shielding / plasma-frequency tests.

Claims checked:
  1. The ionosphere's plasma frequency is ~9 MHz -- it reflects AM radio (below)
     and passes FM/TV (above), and the Debye length there is millimetres.
  2. Many particles sit inside a Debye sphere (N_D >> 1) -- the plasma condition.
  3. lambda_D ~ sqrt(T/n); omega_p ~ sqrt(n).
  4. The critical density inverts the plasma frequency (ionospheric cutoff).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from debye import (debye_length, plasma_parameter, is_plasma,  # noqa: E402
                   plasma_frequency, plasma_frequency_hz, critical_density)


def test_ionosphere_plasma_frequency():
    f = plasma_frequency_hz(1e12) / 1e6  # MHz
    assert abs(f - 9.0) < 1.0, f"ionospheric plasma frequency {f} MHz not ~9"


def test_ionosphere_debye_millimetres():
    lD = debye_length(1e12, 1000.0)
    assert 1e-3 < lD < 1e-2, f"ionospheric Debye length {lD} m not mm-scale"


def test_many_particles_in_debye_sphere():
    assert plasma_parameter(1e12, 1000.0) > 1e3


def test_debye_scaling():
    base = debye_length(1e12, 1000.0)
    # 4x density -> half lambda_D; 4x temperature -> double lambda_D
    assert abs(debye_length(4e12, 1000.0) / base - 0.5) < 1e-9
    assert abs(debye_length(1e12, 4000.0) / base - 2.0) < 1e-9


def test_plasma_frequency_sqrt_n():
    base = plasma_frequency(1e12)
    assert abs(plasma_frequency(4e12) / base - 2.0) < 1e-9


def test_critical_density_inverts():
    n = critical_density(9e6)
    assert abs(plasma_frequency_hz(n) - 9e6) < 1.0


def test_is_plasma_needs_size_and_population():
    # a big ionospheric region is a plasma; a sub-Debye-length blob is not
    assert is_plasma(1e12, 1000.0, 1000.0)
    assert not is_plasma(1e12, 1000.0, 1e-4)


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
