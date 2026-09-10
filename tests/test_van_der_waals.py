"""Van-der-Waals gas tests.

Claims checked:
  1. CO2's critical constants come out right: T_c ~ 304 K, P_c ~ 7.4 MPa.
  2. The critical compressibility P_c V_c / R T_c = 3/8 is universal (a, b cancel).
  3. At large volume the gas approaches ideal (Z -> 1).
  4. The reduced-variable law (P/P_c, V/V_c, T/T_c) gives P_r = 1 at the critical point.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from van_der_waals import (pressure, critical_temperature, critical_volume,  # noqa: E402
                           critical_pressure, critical_compressibility,
                           compressibility_factor, reduced_pressure, R_GAS,
                           A_CO2, B_CO2, A_WATER, B_WATER, A_HELIUM, B_HELIUM)


def test_co2_critical_temperature():
    assert abs(critical_temperature(A_CO2, B_CO2) - 304.0) < 2.0


def test_co2_critical_pressure():
    assert abs(critical_pressure(A_CO2, B_CO2) / 1e6 - 7.4) < 0.2


def test_universal_compressibility():
    for a, b in ((A_CO2, B_CO2), (A_WATER, B_WATER), (A_HELIUM, B_HELIUM)):
        assert abs(critical_compressibility(a, b) - 0.375) < 1e-9


def test_ideal_gas_limit():
    # large volume -> Z -> 1
    assert abs(compressibility_factor(1.0, 10.0, 300.0, A_CO2, B_CO2) - 1.0) < 1e-3


def test_attraction_lowers_Z():
    # near critical density attraction pulls Z below 1
    Vc = critical_volume(B_CO2)
    Z = compressibility_factor(1.0, 2 * Vc, 320.0, A_CO2, B_CO2)
    assert Z < 1.0


def test_reduced_critical_point():
    assert abs(reduced_pressure(1.0, 1.0, 1.0) - 1.0) < 1e-9


def test_critical_volume_three_b():
    assert abs(critical_volume(B_CO2) - 3.0 * B_CO2) < 1e-30


def test_higher_a_higher_Tc():
    # stronger attraction (bigger a) raises the critical temperature
    assert critical_temperature(A_WATER, B_WATER) > critical_temperature(A_HELIUM, B_HELIUM)


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
