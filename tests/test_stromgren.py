"""Stromgren-sphere / HII-region tests.

Claims checked:
  1. An O star in a typical cloud (n ~ 100 /cc) ionizes a ~pc-scale bubble; a
     weaker B star's sphere is much smaller.
  2. R_s ~ Q^(1/3) (photon output) and R_s ~ n^(-2/3) (density).
  3. In equilibrium the enclosed recombination rate equals the ionizing output Q.
  4. The recombination time 1/(n alpha_B) is ~1000 yr at n ~ 100 /cc.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stromgren import (stromgren_radius, stromgren_radius_pc,  # noqa: E402
                       recombination_rate, ionized_mass, recombination_time,
                       Q_O5, Q_B0, ALPHA_B, PC, M_SUN)


def test_o_star_parsec_scale():
    R = stromgren_radius_pc(Q_O5, 100.0)
    assert 1.0 < R < 20.0, f"O-star Stromgren radius {R} pc off scale"


def test_b_star_smaller():
    assert stromgren_radius_pc(Q_B0, 100.0) < stromgren_radius_pc(Q_O5, 100.0)


def test_q_one_third_scaling():
    base = stromgren_radius(Q_O5, 1e8)
    big = stromgren_radius(8 * Q_O5, 1e8)
    assert abs(big / base - 2.0) < 1e-9, "R should scale as Q^(1/3)"


def test_density_two_thirds_scaling():
    base = stromgren_radius(Q_O5, 1e8)
    dense = stromgren_radius(Q_O5, 8e8)
    assert abs(dense / base - 8.0 ** (-2.0 / 3.0)) < 1e-9, "R should scale as n^(-2/3)"


def test_recombination_balances_output():
    n = 1e8
    R = stromgren_radius(Q_O5, n)
    assert abs(recombination_rate(R, n) / Q_O5 - 1.0) < 1e-6


def test_ionized_mass_positive_and_scales():
    n = 1e8
    R = stromgren_radius(Q_O5, n)
    m1 = ionized_mass(R, n) / M_SUN
    assert m1 > 0
    # mass ~ R^3 n ~ Q / (n alpha) , so denser gas -> smaller ionized mass at fixed Q
    R2 = stromgren_radius(Q_O5, 4 * n)
    assert ionized_mass(R2, 4 * n) < ionized_mass(R, n)


def test_recombination_time():
    t_yr = recombination_time(1e8) / 3.15576e7
    assert 500.0 < t_yr < 3000.0, f"recombination time {t_yr} yr off scale"


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
