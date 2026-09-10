"""Poynting-Robertson drag tests.

Claims checked:
  1. beta ~ 1/s: small grains feel strong radiation pressure; sub-blow-out grains
     (beta > 1/2) are unbound and blown straight out.
  2. The blow-out size for silicate grains around the Sun is ~0.1-0.5 micron.
  3. The inspiral time scales as r^2 (distance) and as s (grain size), and micron
     grains at 1 AU fall in on ~kyr timescales -- far less than the solar age.
  4. beta is distance-independent (radiation pressure and gravity both ~1/r^2).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poynting_robertson import (beta, blowout_size, is_blown_out,  # noqa: E402
                                inspiral_time, inspiral_time_years, AU,
                                RHO_DUST, YEAR)


def test_beta_inverse_size():
    # halving the grain size doubles beta
    assert abs(beta(5e-7) / beta(1e-6) - 2.0) < 1e-9


def test_blowout_size_micron_scale():
    s = blowout_size() * 1e6  # in microns
    assert 0.1 < s < 1.0, f"blow-out size {s} um not ~0.1-0.5"


def test_small_grains_blown_out():
    assert is_blown_out(1e-7)       # 0.1 micron: beta > 1/2
    assert not is_blown_out(1e-5)   # 10 micron: bound


def test_blowout_at_beta_half():
    s = blowout_size()
    assert abs(beta(s) - 0.5) < 1e-9


def test_inspiral_r_squared():
    t1 = inspiral_time(AU, 1e-6)
    t2 = inspiral_time(2 * AU, 1e-6)
    assert abs(t2 / t1 - 4.0) < 1e-6, "inspiral time should scale as r^2"


def test_inspiral_grain_size():
    # bigger grain (smaller beta ~ 1/s) spirals in slower, linearly in s
    t1 = inspiral_time(AU, 1e-6)
    t2 = inspiral_time(AU, 2e-6)
    assert abs(t2 / t1 - 2.0) < 1e-6


def test_micron_grain_infall_is_fast():
    yr = inspiral_time_years(AU, 1e-6)
    assert yr < 1e5, f"micron grain at 1 AU should fall in < 100 kyr, got {yr} yr"


def test_beta_distance_independent():
    # beta has no r dependence; the function does not even take r
    b = beta(1e-6)
    assert b > 0
    # inspiral time carries all the distance dependence instead
    assert inspiral_time(AU, 1e-6) < inspiral_time(5 * AU, 1e-6)


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
