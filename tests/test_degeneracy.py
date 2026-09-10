"""Fermi-degeneracy-pressure tests.

Claims checked:
  1. Non-relativistic degeneracy pressure scales as n^{5/3}, ultra-relativistic
     as n^{4/3} -- the softening that creates the Chandrasekhar mass.
  2. White-dwarf electron densities (~1e36 /m^3) are relativistic; ordinary-metal
     electron densities (~1e29 /m^3) are not.
  3. The transition (p_F = m_e c) sits at ~6e35 /m^3, between the two.
  4. Near the transition the two pressure laws are comparable.
  5. The Fermi energy rises with density (n^{2/3}).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from degeneracy import (fermi_momentum, fermi_energy, pressure_nonrel,  # noqa: E402
                        pressure_relativistic, is_relativistic,
                        transition_density, M_E, C)


def test_nonrel_scaling():
    r = pressure_nonrel(2e36) / pressure_nonrel(1e36)
    assert abs(r - 2 ** (5.0 / 3.0)) < 1e-6, "non-rel pressure should scale as n^{5/3}"


def test_rel_scaling():
    r = pressure_relativistic(2e36) / pressure_relativistic(1e36)
    assert abs(r - 2 ** (4.0 / 3.0)) < 1e-6, "rel pressure should scale as n^{4/3}"


def test_white_dwarf_is_relativistic():
    assert is_relativistic(1e36), "white-dwarf electrons should be relativistic"


def test_metal_is_nonrelativistic():
    assert not is_relativistic(1e29), "metal-density electrons should be non-rel"


def test_transition_density():
    n_t = transition_density()
    assert 1e35 < n_t < 1e36, f"transition density {n_t} off"
    # at the transition p_F ~ m_e c
    assert abs(fermi_momentum(n_t) / (M_E * C) - 1.0) < 1e-9


def test_pressures_comparable_at_transition():
    n_t = transition_density()
    ratio = pressure_nonrel(n_t) / pressure_relativistic(n_t)
    assert 0.3 < ratio < 3.0, "the two laws should match order-of-magnitude at the transition"


def test_fermi_energy_rises_with_density():
    assert fermi_energy(2e36) > fermi_energy(1e36)


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
