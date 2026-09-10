"""Saha-equation / cosmic-recombination tests.

Claims checked:
  1. Recombination (ionization fraction -> 0.5) happens at redshift z ~ 1400 and
     temperature ~3700 K -- NOT at the naive kT = 13.6 eV (~158000 K).
  2. The photon-to-baryon ratio is why: the huge photon bath keeps hydrogen
     ionized far below 13.6 eV.
  3. Ionization is essentially complete at high z (early, hot universe) and
     essentially zero by low z (late, cool universe).
  4. The ionization fraction is monotonic in redshift.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from saha import (saha_ionization_fraction, ionization_at_redshift,  # noqa: E402
                  recombination_redshift, naive_ionization_temperature,
                  baryon_density, T0)


def test_recombination_redshift():
    z = recombination_redshift(0.5)
    assert 1100 < z < 1500, f"recombination redshift {z} not ~1100-1400"


def test_recombination_temperature():
    z = recombination_redshift(0.5)
    T = T0 * (1 + z)
    assert 3000 < T < 4500, f"recombination temperature {T} K not ~3700"


def test_far_below_naive_temperature():
    z = recombination_redshift(0.5)
    T = T0 * (1 + z)
    assert T < 0.1 * naive_ionization_temperature(), \
        "recombination is far cooler than kT = 13.6 eV"


def test_ionized_early_neutral_late():
    assert ionization_at_redshift(1600) > 0.9, "hot early universe should be ionized"
    assert ionization_at_redshift(900) < 0.01, "cool late universe should be neutral"


def test_monotonic_in_redshift():
    xs = [ionization_at_redshift(z) for z in (900, 1100, 1300, 1500, 1700)]
    for i in range(1, len(xs)):
        assert xs[i] >= xs[i - 1], "ionization should rise with redshift (hotter)"


def test_baryon_density_scales_as_cube():
    assert abs(baryon_density(1) / baryon_density(0) - 8.0) < 1e-9


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
