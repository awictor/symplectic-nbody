"""Terminal-velocity / drag tests.

Claims checked:
  1. A belly-down skydiver reaches ~50-60 m/s (~120 mph).
  2. A ~2 mm raindrop falls at ~9 m/s and is firmly in the quadratic-drag regime
     (Re >> 1), while a ~10 micron fog droplet is in the Stokes regime (Re << 1).
  3. Stokes terminal velocity rises as r^2; quadratic terminal velocity as sqrt(r).
  4. At terminal velocity, drag exactly balances weight.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from terminal_velocity import (terminal_velocity, drag_force,  # noqa: E402
                               reynolds_number, stokes_velocity, stokes_drag,
                               sphere_terminal_velocity, G_EARTH, RHO_AIR,
                               MU_AIR)


def test_skydiver_speed():
    v = terminal_velocity(75.0, 0.5, 1.0)
    assert 45.0 < v < 60.0, f"skydiver terminal velocity {v} m/s off range"


def test_raindrop_speed_and_regime():
    v = sphere_terminal_velocity(2e-3, 1000.0)
    assert 6.0 < v < 12.0, f"raindrop speed {v} m/s off scale"
    Re = reynolds_number(v, 4e-3)
    assert Re > 100.0, "a raindrop should be in the quadratic-drag regime"


def test_fog_droplet_is_stokes():
    v = stokes_velocity(1e-5, 1000.0)
    Re = reynolds_number(v, 2e-5)
    assert Re < 1.0, f"fog droplet should be viscous-dominated, Re={Re}"
    assert v < 0.05, "fog droplets fall very slowly"


def test_stokes_r_squared():
    v1 = stokes_velocity(1e-5, 1000.0)
    v2 = stokes_velocity(2e-5, 1000.0)
    assert abs(v2 / v1 - 4.0) < 1e-9, "Stokes velocity should scale as r^2"


def test_quadratic_sqrt_r():
    v1 = sphere_terminal_velocity(1e-3, 1000.0)
    v2 = sphere_terminal_velocity(4e-3, 1000.0)
    assert abs(v2 / v1 - 2.0) < 1e-9, "quadratic terminal velocity should scale as sqrt(r)"


def test_drag_balances_weight():
    m, A, C_d = 75.0, 0.5, 1.0
    v = terminal_velocity(m, A, C_d)
    assert abs(drag_force(v, A, C_d) - m * G_EARTH) < 1e-6


def test_stokes_drag_balances_weight():
    r, rho_p = 1e-5, 1000.0
    v = stokes_velocity(r, rho_p)
    weight = 4.0 / 3.0 * math.pi * r ** 3 * (rho_p - RHO_AIR) * G_EARTH
    assert abs(stokes_drag(r, v) - weight) < 1e-18


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
