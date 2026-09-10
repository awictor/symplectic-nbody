"""Semi-empirical-mass-formula tests.

Claims checked:
  1. Iron-56 has ~8.8 MeV/nucleon binding; the binding-per-nucleon curve peaks near
     the iron group (A ~ 56-62).
  2. Binding per nucleon rises for light nuclei and falls for heavy ones (why fusion
     powers up to iron and fission beyond).
  3. The most-stable Z tracks A/2 for light nuclei and drifts to neutron excess for
     heavy ones (U-238 -> Z=92, Pb-208 -> Z=82).
  4. Even-even nuclei get a positive pairing bonus, odd-odd a penalty.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mass_formula import (pairing_term, binding_energy, binding_per_nucleon,  # noqa: E402
                          most_stable_Z, most_stable_Z_continuous)


def test_iron_binding():
    assert abs(binding_per_nucleon(26, 56) - 8.8) < 0.1


def test_peak_near_iron():
    best = max(range(20, 120), key=lambda A: binding_per_nucleon(most_stable_Z(A), A))
    assert 52 <= best <= 66, f"binding peak at A={best} not near iron"


def test_light_rises_heavy_falls():
    assert binding_per_nucleon(2, 4) < binding_per_nucleon(26, 56)   # He < Fe
    assert binding_per_nucleon(92, 238) < binding_per_nucleon(26, 56)  # U < Fe


def test_valley_of_stability():
    assert most_stable_Z(56) == 26          # iron
    assert most_stable_Z(208) == 82         # lead
    assert most_stable_Z(238) == 92         # uranium


def test_neutron_excess_grows():
    # heavy nuclei have Z/A well below 0.5
    assert most_stable_Z(238) / 238.0 < 0.42
    assert abs(most_stable_Z(16) / 16.0 - 0.5) < 0.05   # light ~ Z=N


def test_continuous_matches_integer():
    for A in (56, 120, 208, 238):
        assert abs(most_stable_Z_continuous(A) - most_stable_Z(A)) < 1.5


def test_pairing_signs():
    assert pairing_term(26, 56) > 0.0    # even-even
    assert pairing_term(7, 14) < 0.0     # odd-odd (N-14: Z=7, N=7)
    assert pairing_term(26, 57) == 0.0   # even-odd


def test_binding_positive_for_stable():
    assert binding_energy(26, 56) > 400.0   # MeV, bound


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
