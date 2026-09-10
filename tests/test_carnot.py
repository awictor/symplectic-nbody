"""Carnot-cycle tests.

Claims checked:
  1. eta = 1 - T_c/T_h; a steam plant (800 K -> 300 K) is capped at ~62.5%.
  2. Work plus rejected heat equals the heat input (energy conservation), and heat is
     always rejected (second law).
  3. The heat-pump COP is the fridge COP plus one, and both diverge as the temperature
     gap shrinks.
  4. Over a full reversible cycle the total entropy change is zero.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from carnot import (carnot_efficiency, work_output, heat_rejected,  # noqa: E402
                    cop_refrigerator, cop_heat_pump, entropy_change)


def test_power_plant_efficiency():
    assert abs(carnot_efficiency(800.0, 300.0) - 0.625) < 1e-9


def test_efficiency_below_one():
    for Th, Tc in ((800, 300), (5800, 300), (1000, 1)):
        assert 0.0 < carnot_efficiency(Th, Tc) < 1.0


def test_energy_conservation():
    Qh = 1000.0
    W = work_output(Qh, 800.0, 300.0)
    Qc = heat_rejected(Qh, 800.0, 300.0)
    assert abs(W + Qc - Qh) < 1e-9


def test_heat_always_rejected():
    assert heat_rejected(1000.0, 800.0, 300.0) > 0.0


def test_heat_pump_is_fridge_plus_one():
    assert abs(cop_heat_pump(293.0, 273.0) - cop_refrigerator(293.0, 273.0) - 1.0) < 1e-9


def test_cop_diverges_small_gap():
    big_gap = cop_heat_pump(400.0, 300.0)
    small_gap = cop_heat_pump(310.0, 300.0)
    assert small_gap > big_gap


def test_reversible_cycle_zero_entropy():
    Qh = 1000.0
    Qc = heat_rejected(Qh, 800.0, 300.0)
    dS = -entropy_change(Qh, 800.0) + entropy_change(Qc, 300.0)
    assert abs(dS) < 1e-9


def test_hotter_source_more_efficient():
    assert carnot_efficiency(1000.0, 300.0) > carnot_efficiency(500.0, 300.0)


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
