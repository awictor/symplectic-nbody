"""Radioactive-decay tests.

Claims checked:
  1. Exponential decay: half remains after one half-life; the decay constant is
     ln2/t_half.
  2. Carbon-14 dating: 25% remaining is two half-lives ~11,460 yr.
  3. The Bateman daughter starts at zero, rises to a peak, then falls.
  4. In secular equilibrium the daughter activity equals the parent activity,
     independent of the daughter's (short) half-life.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from radioactive_decay import (decay_constant, remaining, activity,  # noqa: E402
                               age_from_fraction, n_half_lives,
                               bateman_daughter, secular_equilibrium_activity,
                               LN2)

T_C14 = 5730.0


def test_half_after_one_half_life():
    assert abs(remaining(100.0, T_C14, T_C14) - 50.0) < 1e-9


def test_decay_constant():
    assert abs(decay_constant(T_C14) - LN2 / T_C14) < 1e-30


def test_c14_dating():
    assert abs(age_from_fraction(0.25, T_C14) - 11460.0) < 1.0


def test_n_half_lives():
    assert abs(n_half_lives(0.125) - 3.0) < 1e-9   # 1/8 = 3 half-lives


def test_activity_proportional_to_N():
    assert abs(activity(200.0, T_C14) / activity(100.0, T_C14) - 2.0) < 1e-9


def test_bateman_rises_then_falls():
    # daughter (short-lived) peaks then decays; sample early, peak, late
    early = bateman_daughter(1000.0, 0.1, 8.0, 0.25)
    peak = bateman_daughter(1000.0, 1.0, 8.0, 0.25)
    late = bateman_daughter(1000.0, 40.0, 8.0, 0.25)
    assert early < peak
    assert late < peak


def test_bateman_starts_zero():
    assert abs(bateman_daughter(1000.0, 0.0, 8.0, 0.25)) < 1e-9


def test_secular_equilibrium():
    # daughter activity = parent activity regardless of daughter half-life
    N_p = 1e12
    A = secular_equilibrium_activity(N_p, 1e6)
    assert abs(A - activity(N_p, 1e6)) < 1e-6


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
