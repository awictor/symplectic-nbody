"""Bondi-accretion tests.

Claims checked:
  1. The Bondi rate scales as M^2 (runaway growth), linearly with ambient
     density, and as c_s^{-3} (cold gas accretes far faster).
  2. The Bondi radius r_B = G M / c_s^2 grows with mass and shrinks with sound
     speed, and is ~tens of AU for a 10 M_sun object in warm ISM.
  3. A 10 M_sun object in warm ISM accretes at ~1e-13 M_sun/yr -- a small rate,
     which is why isolated stellar-mass black holes are hard to see.
  4. The accretion luminosity is eta Mdot c^2.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bondi import (bondi_radius, bondi_rate, accretion_luminosity,  # noqa: E402
                   sound_speed, M_SUN, YEAR, C)

CS = sound_speed(1e4, mu=1.0)
RHO = 1.67e-21
M10 = 10 * M_SUN


def test_mass_squared_scaling():
    assert abs(bondi_rate(2 * M10, RHO, CS) / bondi_rate(M10, RHO, CS) - 4.0) < 1e-9


def test_density_linear():
    assert abs(bondi_rate(M10, 2 * RHO, CS) / bondi_rate(M10, RHO, CS) - 2.0) < 1e-9


def test_sound_speed_cubed():
    r = bondi_rate(M10, RHO, 2 * CS) / bondi_rate(M10, RHO, CS)
    assert abs(r - 0.125) < 1e-9, "rate should scale as c_s^{-3}"


def test_bondi_radius_scale():
    rb = bondi_radius(M10, CS) / 1.496e11  # AU
    assert 20 < rb < 200, f"Bondi radius {rb} AU off"
    assert bondi_radius(2 * M10, CS) > bondi_radius(M10, CS)
    assert bondi_radius(M10, 2 * CS) < bondi_radius(M10, CS)


def test_accretion_rate_scale():
    mdot = bondi_rate(M10, RHO, CS) / M_SUN * YEAR
    assert 1e-15 < mdot < 1e-12, f"Bondi rate {mdot} Msun/yr off scale"


def test_accretion_luminosity():
    mdot = bondi_rate(M10, RHO, CS)
    assert abs(accretion_luminosity(mdot, 0.1) - 0.1 * mdot * C * C) < 1e-6 * accretion_luminosity(mdot, 0.1)


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
