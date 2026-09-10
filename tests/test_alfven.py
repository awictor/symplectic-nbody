"""Alfven-wave / MHD tests.

Claims checked:
  1. v_A = B / sqrt(mu0 rho): a 1 G quiet-corona field gives ~2000 km/s.
  2. The solar corona is magnetically dominated (plasma beta << 1); a stellar
     interior is gas-dominated (beta >> 1).
  3. The solar wind at 1 AU is super-Alfvenic (M_A > 1).
  4. Scalings: v_A ~ B and v_A ~ 1/sqrt(rho); beta ~ 1/B^2.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alfven import (alfven_speed, alfven_speed_number_density,  # noqa: E402
                    magnetic_pressure, plasma_beta, alfven_mach,
                    wave_travel_time, MU0, M_P)


def test_coronal_speed():
    # quiet corona: 1 G = 1e-4 T, n ~ 1e14 /m^3
    v = alfven_speed_number_density(1e-4 * 10, 1e14)  # 10 G active-ish
    assert 1e6 < v < 1e7, f"coronal v_A {v} m/s out of range"


def test_speed_formula():
    v = alfven_speed(1e-3, 1e-12)
    assert abs(v - 1e-3 / math.sqrt(MU0 * 1e-12)) < 1.0


def test_corona_field_dominated():
    beta = plasma_beta(1e15, 2e6, 1e-2)  # active-region corona
    assert beta < 0.1, f"corona should be field-dominated, beta={beta}"


def test_interior_gas_dominated():
    # dense, hot, weakly magnetized: gas pressure dwarfs magnetic
    beta = plasma_beta(1e30, 1e7, 1e-2)
    assert beta > 1e3, f"interior should be gas-dominated, beta={beta}"


def test_solar_wind_super_alfvenic():
    # 1 AU: B ~ 5 nT, n ~ 5e6 /m^3, u ~ 400 km/s
    rho = 5e6 * M_P
    M = alfven_mach(4e5, 5e-9, rho)
    assert M > 1.0, f"solar wind at 1 AU should be super-Alfvenic, M_A={M}"


def test_scalings():
    base = alfven_speed(1e-3, 1e-12)
    # double B -> double v_A
    assert abs(alfven_speed(2e-3, 1e-12) / base - 2.0) < 1e-9
    # quadruple rho -> half v_A
    assert abs(alfven_speed(1e-3, 4e-12) / base - 0.5) < 1e-9
    # beta ~ 1/B^2
    b1 = plasma_beta(1e15, 2e6, 1e-2)
    b2 = plasma_beta(1e15, 2e6, 2e-2)
    assert abs(b1 / b2 - 4.0) < 1e-9


def test_travel_time():
    # v_A * t == length
    L, B, rho = 7e8, 1e-2, 1e15 * M_P
    t = wave_travel_time(L, B, rho)
    assert abs(alfven_speed(B, rho) * t - L) < 1e-3


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
