"""Reynolds-number / pipe-flow tests.

Claims checked:
  1. Pipe flow transitions near Re ~ 2300 (laminar below, turbulent above ~4000).
  2. Re = rho v L / mu; a bacterium lives at Re ~ 1e-5 (pure viscosity), a whale at
     Re ~ 1e8 (pure inertia).
  3. Hagen-Poiseuille laminar flow scales as radius^4.
  4. The kinematic and dynamic-viscosity forms agree; the critical velocity inverts.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reynolds import (reynolds_number, reynolds_kinematic, flow_regime,  # noqa: E402
                      is_turbulent, hagen_poiseuille_flow, critical_velocity,
                      entrance_length, RHO_WATER, MU_WATER, NU_WATER,
                      RE_LAMINAR)


def test_laminar_below_2300():
    Re = reynolds_number(RHO_WATER, 0.1, 0.02, MU_WATER)
    assert Re < RE_LAMINAR
    assert flow_regime(Re) == "laminar"


def test_turbulent_at_high_speed():
    Re = reynolds_number(RHO_WATER, 1.0, 0.02, MU_WATER)
    assert is_turbulent(Re)
    assert flow_regime(Re) == "turbulent"


def test_transitional_band():
    assert flow_regime(3000.0) == "transitional"


def test_kinematic_matches_dynamic():
    a = reynolds_number(RHO_WATER, 0.5, 0.01, MU_WATER)
    b = reynolds_kinematic(0.5, 0.01, NU_WATER)
    assert abs(a / b - 1.0) < 1e-9


def test_bacterium_and_whale():
    bac = reynolds_number(RHO_WATER, 30e-6, 1e-6, MU_WATER)
    whale = reynolds_number(RHO_WATER, 10.0, 20.0, MU_WATER)
    assert bac < 1e-3
    assert whale > 1e7


def test_poiseuille_r_fourth():
    q1 = hagen_poiseuille_flow(1000.0, 0.01, 1.0, MU_WATER)
    q2 = hagen_poiseuille_flow(1000.0, 0.02, 1.0, MU_WATER)
    assert abs(q2 / q1 - 16.0) < 1e-6


def test_critical_velocity_inverts():
    v = critical_velocity(0.02, RHO_WATER, MU_WATER)
    assert abs(reynolds_number(RHO_WATER, v, 0.02, MU_WATER) - RE_LAMINAR) < 1e-6


def test_entrance_length_scales():
    assert abs(entrance_length(2000.0, 0.02) - 0.05 * 2000.0 * 0.02) < 1e-9


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
