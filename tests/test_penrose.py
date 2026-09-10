"""Penrose-process / black-hole-energetics tests.

Claims checked:
  1. A non-spinning (a=0) hole has M_irr = M and zero extractable energy.
  2. An extremal (a=M) hole has M_irr = M/sqrt(2) and 29.3% of its mass-energy is
     rotational and extractable -- the maximum.
  3. More spin means more extractable rotational energy (monotonic in a).
  4. Extracting spin (a -> 0 at fixed M) GROWS the irreducible mass and the
     horizon area -- Hawking's area theorem; the process cannot reduce the area.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from penrose import (irreducible_mass, rotational_energy,  # noqa: E402
                     rotational_energy_fraction, max_efficiency_extremal,
                     area_irreducible)


def test_no_spin_no_energy():
    assert abs(irreducible_mass(1.0, 0.0) - 1.0) < 1e-12
    assert abs(rotational_energy(1.0, 0.0)) < 1e-12


def test_extremal_irreducible_mass():
    assert abs(irreducible_mass(1.0, 1.0) - 1.0 / math.sqrt(2.0)) < 1e-9


def test_max_extractable_fraction():
    frac = max_efficiency_extremal()
    assert abs(frac - 0.2929) < 1e-3, f"extremal extractable fraction {frac} not ~0.293"


def test_more_spin_more_energy():
    e_lo = rotational_energy_fraction(1.0, 0.3)
    e_hi = rotational_energy_fraction(1.0, 0.9)
    assert 0.0 < e_lo < e_hi, "higher spin should have more extractable energy"


def test_area_theorem():
    # extracting spin (a -> 0 at fixed total M) grows the irreducible mass & area
    a_hi = area_irreducible(1.0, 1.0)
    a_mid = area_irreducible(1.0, 0.5)
    a_lo = area_irreducible(1.0, 0.0)
    assert a_hi < a_mid < a_lo, "irreducible-mass area must grow as spin is removed"


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
