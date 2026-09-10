"""Big Bang nucleosynthesis tests.

Claims checked:
  1. The equilibrium n/p ratio approaches 1 at high temperature (>> 1 MeV) and
     falls as the universe cools.
  2. At weak freeze-out (~0.8 MeV) the ratio locks in near 1/6-1/5.
  3. Free-neutron decay before capture lowers n/p to ~1/7.
  4. The primordial helium mass fraction Y_p comes out ~0.25 -- the observed
     value, and one of the strongest confirmations of the hot Big Bang.
  5. Longer decay time (later nucleosynthesis) yields less helium.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bbn import (np_ratio_equilibrium, np_ratio_at_freezeout,  # noqa: E402
                 np_ratio_after_decay, helium_mass_fraction, TAU_N)


def test_equilibrium_ratio_limits():
    assert np_ratio_equilibrium(100.0) > 0.98, "n/p -> 1 when very hot"
    assert np_ratio_equilibrium(0.1) < 0.01, "n/p tiny when cold"
    # monotonic: cooler -> fewer neutrons
    assert np_ratio_equilibrium(2.0) > np_ratio_equilibrium(1.0)


def test_freezeout_ratio():
    r = np_ratio_at_freezeout()
    assert 0.15 < r < 0.22, f"freeze-out n/p {r} not ~1/6-1/5"


def test_decay_lowers_ratio():
    assert np_ratio_after_decay() < np_ratio_at_freezeout(), "decay reduces n/p"
    assert 0.12 < np_ratio_after_decay() < 0.17, "post-decay n/p should be ~1/7"


def test_helium_fraction():
    Y = helium_mass_fraction()
    assert 0.22 < Y < 0.28, f"primordial helium Y_p {Y} not ~0.25"


def test_later_nucleosynthesis_less_helium():
    early = helium_mass_fraction(150.0)
    late = helium_mass_fraction(400.0)
    assert late < early, "more neutron decay before capture -> less helium"


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
