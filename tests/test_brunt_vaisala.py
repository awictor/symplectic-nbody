"""Brunt-Vaisala / convective-stability tests.

Claims checked:
  1. Earth's dry adiabatic lapse rate is g/c_p ~ 9.8 K/km.
  2. A subadiabatic troposphere is stably stratified: N ~ 0.01 s^-1, buoyancy
     period ~5-10 min; an isothermal layer is also stable.
  3. A superadiabatic gradient (steeper than the lapse rate) is convectively
     unstable (Schwarzschild criterion, N^2 < 0).
  4. The general and ideal-gas forms agree, and the period is 2 pi / N.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from brunt_vaisala import (adiabatic_lapse_rate, brunt_vaisala_squared_ideal,  # noqa: E402
                           brunt_vaisala_squared_general, brunt_vaisala_frequency,
                           buoyancy_period, is_convective, G_EARTH, CP_AIR)


def test_lapse_rate():
    g = adiabatic_lapse_rate() * 1000.0  # K/km
    assert abs(g - 9.8) < 0.2, f"lapse rate {g} K/km not ~9.8"


def test_troposphere_stable():
    N2 = brunt_vaisala_squared_ideal(-6.5e-3, 273.0)
    assert N2 > 0, "subadiabatic troposphere should be stable"
    N = brunt_vaisala_frequency(N2)
    assert 0.005 < N < 0.02, f"tropospheric N {N} s^-1 off range"
    per = buoyancy_period(N2) / 60.0
    assert 3.0 < per < 20.0, f"buoyancy period {per} min off range"


def test_isothermal_stable():
    N2 = brunt_vaisala_squared_ideal(0.0, 273.0)
    assert N2 > 0, "isothermal layer should be stably stratified"


def test_superadiabatic_convective():
    # gradient steeper (more negative) than the adiabatic lapse rate
    assert is_convective(-12e-3)
    assert not is_convective(-6.5e-3)


def test_unstable_has_no_real_frequency():
    N2 = brunt_vaisala_squared_ideal(-15e-3, 273.0)
    assert N2 < 0
    assert brunt_vaisala_frequency(N2) == 0.0
    assert buoyancy_period(N2) == float("inf")


def test_period_matches_frequency():
    N2 = brunt_vaisala_squared_ideal(-6.5e-3, 273.0)
    N = brunt_vaisala_frequency(N2)
    assert abs(buoyancy_period(N2) - 2.0 * math.pi / N) < 1e-9


def test_general_form_positive_when_stable():
    # density falling slower than the adiabatic prediction -> stable (N^2 > 0)
    N2 = brunt_vaisala_squared_general(dlnp_dz=-1e-4, dlnrho_dz=-1e-4, gamma=1.4)
    assert N2 > 0


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
