"""Supernova-remnant phase tests.

Claims checked:
  1. Free expansion ends when the swept-up mass equals the ejecta mass, at a few
     parsecs after a few hundred years.
  2. The Sedov phase gives R ~ t^(2/5), reaching ~5 pc by 1000 yr and slowing from
     thousands of km/s.
  3. The four phases (free expansion -> Sedov -> snowplow -> merged) are named in
     the right time order.
  4. The remnant merges into the ISM at ~100 pc scale when the shock slows to
     ~10 km/s.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snr_phases import (ism_density, sweep_up_radius,  # noqa: E402
                        free_expansion_end_time, sedov_radius, sedov_velocity,
                        phase, merge_radius, PC, M_SUN, YEAR, E_SN)

RHO = ism_density(1.0)
M_EJ = 5 * M_SUN
V_EJ = 1e7  # 10,000 km/s


def test_sweep_up_radius_few_pc():
    R = sweep_up_radius(M_EJ, RHO) / PC
    assert 1.0 < R < 10.0, f"sweep-up radius {R} pc off scale"


def test_free_expansion_ends_early():
    t = free_expansion_end_time(M_EJ, RHO, V_EJ) / YEAR
    assert 50.0 < t < 2000.0, f"free-expansion end {t} yr off scale"


def test_sedov_two_fifths():
    r1 = sedov_radius(1000 * YEAR)
    r2 = sedov_radius(4000 * YEAR)
    assert abs(r2 / r1 - 4.0 ** 0.4) < 1e-6, "Sedov R should scale as t^(2/5)"


def test_sedov_radius_at_1000yr():
    R = sedov_radius(1000 * YEAR) / PC
    assert 3.0 < R < 8.0, f"Sedov radius at 1000 yr {R} pc off scale"


def test_sedov_decelerates():
    assert sedov_velocity(5000 * YEAR) < sedov_velocity(1000 * YEAR)


def test_phase_ordering():
    assert phase(100 * YEAR, M_EJ, V_EJ) == "free expansion"
    assert phase(2000 * YEAR, M_EJ, V_EJ) == "Sedov-Taylor"
    assert phase(5e4 * YEAR, M_EJ, V_EJ) == "snowplow"
    assert phase(2e6 * YEAR, M_EJ, V_EJ) == "merged"


def test_merge_radius_scale():
    R = merge_radius() / PC
    assert 50.0 < R < 300.0, f"merge radius {R} pc off scale"


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
