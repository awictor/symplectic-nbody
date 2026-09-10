"""Tidal-heating tests.

Claims checked:
  1. Io's tidal heating comes out ~1e14 W, matching the observed heat output that
     powers its volcanoes.
  2. Its surface heat flux is a few W/m^2 -- far above Earth's ~0.08 W/m^2.
  3. The heating scales as e^2, as R^5, and as a^{-15/2} (via n/a^6).
  4. A circular orbit (e=0) dissipates no tidal heat.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tidal_heating import (tidal_heating_rate, io_heating, surface_heat_flux,  # noqa: E402
                           M_JUP, IO_R, IO_A, IO_E, IO_K2_OVER_Q, IO_AREA)


def test_io_heating_scale():
    p = io_heating()
    assert 3e13 < p < 3e14, f"Io heating {p} W not ~1e14"


def test_io_surface_flux():
    flux = surface_heat_flux(io_heating(), IO_AREA)
    assert 1.0 < flux < 5.0, f"Io surface flux {flux} W/m^2 not a few W/m^2"
    assert flux > 0.08, "Io should far exceed Earth's ~0.08 W/m^2"


def test_e_squared_scaling():
    base = io_heating()
    doubled = tidal_heating_rate(M_JUP, IO_R, IO_A, 2 * IO_E, IO_K2_OVER_Q)
    assert abs(doubled / base - 4.0) < 1e-9, "heating should scale as e^2"


def test_r_fifth_scaling():
    base = io_heating()
    bigger = tidal_heating_rate(M_JUP, 2 * IO_R, IO_A, IO_E, IO_K2_OVER_Q)
    assert abs(bigger / base - 32.0) < 1e-9, "heating should scale as R^5"


def test_semi_major_axis_scaling():
    base = io_heating()
    farther = tidal_heating_rate(M_JUP, IO_R, 2 * IO_A, IO_E, IO_K2_OVER_Q)
    assert abs(farther / base - 2 ** (-7.5)) < 1e-6, "heating should scale as a^{-15/2}"


def test_circular_orbit_no_heating():
    assert tidal_heating_rate(M_JUP, IO_R, IO_A, 0.0, IO_K2_OVER_Q) == 0.0


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
