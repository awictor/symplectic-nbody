"""Sound-speed / Rankine-Hugoniot shock-jump tests.

Claims checked:
  1. The sound speed in sea-level air (288 K, diatomic) is ~340 m/s.
  2. A Mach-1 "shock" is the identity: no jump in density, pressure or temperature.
  3. Density compression saturates at 4 (gamma=5/3) while pressure and temperature
     jumps grow without bound as M^2.
  4. The post-shock flow is always subsonic (M2 < 1) for a real shock (M1 > 1).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shock_jump import (sound_speed, sound_speed_from_pressure,  # noqa: E402
                        density_ratio, pressure_ratio, temperature_ratio,
                        downstream_mach, strong_shock_compression,
                        GAMMA_MONO)


def test_air_sound_speed():
    c = sound_speed(288.0, mu=28.97, gamma=1.4)
    assert abs(c - 340.0) < 5.0, f"air sound speed {c} m/s not ~340"


def test_pressure_form_agrees():
    # c_s = sqrt(gamma P / rho); build P from ideal gas and check consistency
    T, mu = 1e4, 0.6
    rho = 1e-20
    P = rho / (mu * 1.6726219e-27) * 1.380649e-23 * T
    c1 = sound_speed(T, mu)
    c2 = sound_speed_from_pressure(P, rho)
    assert abs(c1 / c2 - 1.0) < 1e-6


def test_mach_one_identity():
    assert abs(density_ratio(1.0) - 1.0) < 1e-9
    assert abs(pressure_ratio(1.0) - 1.0) < 1e-9
    assert abs(temperature_ratio(1.0) - 1.0) < 1e-9
    assert abs(downstream_mach(1.0) - 1.0) < 1e-9


def test_compression_saturates_at_four():
    assert abs(strong_shock_compression() - 4.0) < 1e-12
    assert density_ratio(1e6) < 4.0 + 1e-6
    assert density_ratio(1e6) > 3.99


def test_pressure_grows_unbounded():
    # pressure jump ~ M^2: 10x the Mach -> ~100x the pressure ratio at high M
    assert abs(pressure_ratio(100.0) / pressure_ratio(10.0) - 100.0) < 1.0


def test_temperature_rises_with_mach():
    assert temperature_ratio(20.0) > temperature_ratio(5.0) > temperature_ratio(2.0)


def test_downstream_subsonic():
    for M in (1.5, 2, 5, 10, 100):
        assert downstream_mach(M) < 1.0, f"post-shock flow at M={M} should be subsonic"


def test_stronger_shock_hotter_and_denser():
    assert density_ratio(5.0) > density_ratio(2.0)
    assert pressure_ratio(5.0) > pressure_ratio(2.0)


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
