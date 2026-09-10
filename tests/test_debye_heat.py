"""Debye-specific-heat tests.

Claims checked:
  1. High temperature (T >> Theta_D) recovers the Dulong-Petit value 3R.
  2. Low temperature (T << Theta_D) follows the Debye T^3 law, matching the closed-form
     limit.
  3. Copper (Theta_D=343 K) is ~94% of 3R at room temperature; diamond (Theta_D=2230 K)
     is still far below it -- stiff light lattices stay "cold".
  4. The heat capacity rises monotonically with temperature toward 3R.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from debye_heat import (heat_capacity, low_temperature_limit,  # noqa: E402
                        high_temperature_limit, fraction_of_dulong_petit,
                        DULONG_PETIT, R_GAS, THETA_COPPER, THETA_DIAMOND, THETA_LEAD)


def test_high_temperature_dulong_petit():
    C = heat_capacity(10 * THETA_COPPER, THETA_COPPER)
    assert abs(C - DULONG_PETIT) < 0.1


def test_low_temperature_t_cubed():
    T = THETA_COPPER / 25.0   # well below Theta_D
    full = heat_capacity(T, THETA_COPPER)
    approx = low_temperature_limit(T, THETA_COPPER)
    assert abs(full / approx - 1.0) < 0.02


def test_t_cubed_scaling():
    lo = low_temperature_limit(10.0, THETA_COPPER)
    hi = low_temperature_limit(20.0, THETA_COPPER)
    assert abs(lo / hi - 0.125) < 1e-9   # (1/2)^3


def test_copper_room_temperature():
    assert abs(fraction_of_dulong_petit(300.0, THETA_COPPER) - 0.94) < 0.02


def test_diamond_still_cold():
    # diamond's high Theta_D keeps its heat capacity well below 3R at 300 K
    assert heat_capacity(300.0, THETA_DIAMOND) < 0.3 * DULONG_PETIT


def test_monotonic_increase():
    c1 = heat_capacity(50.0, THETA_COPPER)
    c2 = heat_capacity(150.0, THETA_COPPER)
    c3 = heat_capacity(400.0, THETA_COPPER)
    assert c1 < c2 < c3


def test_high_limit_value():
    assert abs(high_temperature_limit() - 3.0 * R_GAS) < 1e-9


def test_lower_theta_warms_faster():
    # at fixed T, a lower Debye temperature is closer to 3R (lead vs diamond)
    assert (fraction_of_dulong_petit(300.0, THETA_LEAD)
            > fraction_of_dulong_petit(300.0, THETA_DIAMOND))


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
