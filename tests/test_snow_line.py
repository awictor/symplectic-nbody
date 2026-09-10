"""Snow-line / disk-temperature tests.

Claims checked:
  1. The equilibrium disk temperature at 1 AU is ~280 K, and T ~ r^(-1/2).
  2. The water snow line of a solar disk sits at ~2.7-3.1 AU (between Mars and
     Jupiter), and CO2/CO frost lines lie much farther out.
  3. The snow line moves outward as sqrt(L): a brighter star pushes ice outward.
  4. Ice condensation roughly triples the solids surface density across the line.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snow_line import (disk_temperature, snow_line, snow_line_au,  # noqa: E402
                       solids_density_jump, L_SUN, AU, T_WATER_ICE,
                       T_CO2_ICE, T_CO_ICE, ICE_ENHANCEMENT)


def test_temperature_at_1au():
    T = disk_temperature(AU)
    assert abs(T - 278.0) < 5.0, f"disk T at 1 AU {T} K not ~278"


def test_inverse_sqrt_r():
    # quadruple the distance -> halve the temperature
    assert abs(disk_temperature(4 * AU) / disk_temperature(AU) - 0.5) < 1e-9


def test_water_snow_line_location():
    a = snow_line_au()
    assert 2.5 < a < 3.2, f"water snow line {a} AU not between Mars and Jupiter"


def test_frost_line_ordering():
    # colder condensation temperatures freeze out farther from the star
    assert snow_line_au(T_WATER_ICE) < snow_line_au(T_CO2_ICE) < snow_line_au(T_CO_ICE)


def test_sqrt_luminosity_scaling():
    near = snow_line(T_WATER_ICE, L_SUN)
    far = snow_line(T_WATER_ICE, 4 * L_SUN)
    assert abs(far / near - 2.0) < 1e-9, "snow line should scale as sqrt(L)"


def test_solids_jump():
    assert abs(solids_density_jump(10.0) - 10.0 * ICE_ENHANCEMENT) < 1e-9


def test_temperature_snowline_consistency():
    # by construction, T at the snow line equals the condensation temperature
    r = snow_line(T_WATER_ICE)
    assert abs(disk_temperature(r) - T_WATER_ICE) < 1e-6


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
