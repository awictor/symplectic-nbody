"""Sedov-Taylor blast-wave tests.

Claims checked:
  1. Self-similar scaling: R ~ t^{2/5} and v ~ t^{-3/5}.
  2. R(t) and energy_from_radius are exact inverses -- so an observed radius and
     time recover the explosion energy (Taylor's Trinity trick), and the
     recovered E is independent of which (R,t) point on the same blast you use.
  3. A ~1e44 J supernova in the ISM makes a parsec-scale remnant expanding at
     ~thousands of km/s, decelerating over time.
  4. The Trinity fireball (~130 m at 25 ms in air) recovers a yield of order
     10 kilotons -- the right order of magnitude for the 21 kt device.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sedov import (shock_radius, shock_velocity, energy_from_radius,  # noqa: E402
                   shock_temperature, swept_mass)

E_SN, RHO_ISM = 1e44, 2e-21
PC = 3.086e16
KT = 4.184e12  # joules per kiloton TNT


def test_radius_scales_as_t_two_fifths():
    r1 = shock_radius(E_SN, RHO_ISM, 1e10)
    r2 = shock_radius(E_SN, RHO_ISM, 2e10)
    assert abs(math.log(r2 / r1) / math.log(2) - 0.4) < 1e-6


def test_velocity_scales_as_t_minus_three_fifths():
    v1 = shock_velocity(E_SN, RHO_ISM, 1e10)
    v2 = shock_velocity(E_SN, RHO_ISM, 2e10)
    assert abs(math.log(v2 / v1) / math.log(2) - (-0.6)) < 1e-6


def test_energy_inversion_is_exact():
    t = 5e10
    R = shock_radius(E_SN, RHO_ISM, t)
    assert abs(energy_from_radius(R, RHO_ISM, t) - E_SN) / E_SN < 1e-9


def test_energy_independent_of_time_point():
    # same blast sampled at two times -> same recovered energy
    R1, t1 = shock_radius(E_SN, RHO_ISM, 1e10), 1e10
    R2, t2 = shock_radius(E_SN, RHO_ISM, 8e10), 8e10
    e1 = energy_from_radius(R1, RHO_ISM, t1)
    e2 = energy_from_radius(R2, RHO_ISM, t2)
    assert abs(e1 - e2) / e1 < 1e-9


def test_supernova_remnant_scale():
    R = shock_radius(E_SN, RHO_ISM, 4e10)   # ~1300 yr
    v = shock_velocity(E_SN, RHO_ISM, 4e10)
    assert 1.0 < R / PC < 30.0, f"SNR radius {R/PC} pc off scale"
    assert 100e3 < v < 1e7, f"SNR shock speed {v} m/s off scale"


def test_trinity_yield_order_of_magnitude():
    kt = energy_from_radius(130.0, 1.25, 0.025) / KT
    assert 3.0 < kt < 40.0, f"Trinity yield {kt} kt not the right order"


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
