"""Greenhouse-effect tests.

Claims checked:
  1. Earth's equilibrium (skin) temperature is ~255 K.
  2. A grey atmosphere warms the surface above equilibrium; Earth's tau ~ 0.84
     gives the observed ~288 K (a ~33 K greenhouse).
  3. Venus needs a huge optical depth (tau ~ 100+) to reach its ~737 K surface --
     the runaway greenhouse.
  4. Surface temperature rises monotonically with optical depth; tau=0 recovers
     the airless equilibrium value.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from greenhouse import (equilibrium_temperature, surface_temperature,  # noqa: E402
                        greenhouse_warming, optical_depth_for_surface,
                        surface_temperature_planet)


def test_earth_equilibrium():
    T = equilibrium_temperature(1.0, 1.0, 0.3)
    assert abs(T - 255.0) < 3.0, f"Earth T_eq {T} not ~255"


def test_earth_surface_33k_greenhouse():
    Te = equilibrium_temperature(1.0, 1.0, 0.3)
    Ts = surface_temperature(Te, 0.84)
    assert abs(Ts - 288.0) < 3.0, f"Earth surface {Ts} not ~288"
    assert abs(greenhouse_warming(Te, 0.84) - 33.0) < 4.0


def test_tau_zero_is_equilibrium():
    Te = equilibrium_temperature(1.0, 1.0, 0.3)
    assert abs(surface_temperature(Te, 0.0) - Te) < 1e-9


def test_venus_runaway():
    Tv = equilibrium_temperature(1.0, 0.72, 0.77)
    tau = optical_depth_for_surface(Tv, 737.0)
    assert tau > 100.0, f"Venus needs tau>100 for its surface, got {tau}"
    # and that optical depth reproduces the surface temperature
    assert abs(surface_temperature(Tv, tau) - 737.0) < 1.0


def test_monotonic_in_tau():
    Te = 255.0
    assert surface_temperature(Te, 2.0) > surface_temperature(Te, 0.5) > Te


def test_optical_depth_inverts_surface():
    Te = 255.0
    tau = optical_depth_for_surface(Te, 300.0)
    assert abs(surface_temperature(Te, tau) - 300.0) < 1e-6


def test_planet_convenience():
    Ts = surface_temperature_planet(1.0, 1.0, 0.3, 0.84)
    assert abs(Ts - 288.0) < 3.0


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
