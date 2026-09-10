"""Bi-elliptic-transfer tests.

Claims checked:
  1. Below the R = 11.94 crossover (e.g. R=10) the Hohmann transfer is always cheaper,
     for any bi-elliptic detour radius.
  2. Above R = 15.58 (e.g. R=16) the bi-elliptic transfer beats Hohmann for a large
     enough detour radius.
  3. In the intermediate band (11.94 < R < 15.58) bi-elliptic wins only with a very
     large detour radius.
  4. The Hohmann delta-v matches the standard vis-viva two-burn result.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bi_elliptic import (hohmann_delta_v, bi_elliptic_delta_v,  # noqa: E402
                         bi_elliptic_is_cheaper, crossover_ratio, MU_EARTH)

R1 = 7000e3


def test_below_crossover_hohmann_wins():
    r2 = 10.0 * R1
    for rb in (20 * R1, 100 * R1, 1e4 * R1):
        assert not bi_elliptic_is_cheaper(R1, r2, rb)


def test_above_crossover_bielliptic_wins():
    r2 = 16.0 * R1
    assert bi_elliptic_is_cheaper(R1, r2, 100 * R1)


def test_intermediate_needs_large_detour():
    r2 = 12.0 * R1
    assert not bi_elliptic_is_cheaper(R1, r2, 50 * R1)   # small detour: Hohmann wins
    assert bi_elliptic_is_cheaper(R1, r2, 1e4 * R1)      # huge detour: bi wins


def test_crossover_value():
    assert abs(crossover_ratio() - 11.94) < 0.01


def test_hohmann_matches_visviva():
    r2 = 5.0 * R1
    mu = MU_EARTH
    v1 = math.sqrt(mu / R1)
    v2 = math.sqrt(mu / r2)
    a_t = 0.5 * (R1 + r2)
    dv = (math.sqrt(mu * (2 / R1 - 1 / a_t)) - v1) + (v2 - math.sqrt(mu * (2 / r2 - 1 / a_t)))
    assert abs(hohmann_delta_v(R1, r2) - dv) < 1e-6


def test_delta_v_positive():
    assert hohmann_delta_v(R1, 8 * R1) > 0
    assert bi_elliptic_delta_v(R1, 8 * R1, 40 * R1) > 0


def test_larger_detour_costs_less_when_bielliptic_favoured():
    # for a big ratio, a larger detour approaches the limiting (cheapest) bi-elliptic
    r2 = 20.0 * R1
    assert bi_elliptic_delta_v(R1, r2, 1e5 * R1) < bi_elliptic_delta_v(R1, r2, 25 * R1)


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
