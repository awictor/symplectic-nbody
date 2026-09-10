"""Fermi-acceleration / diffusive-shock tests.

Claims checked:
  1. Strong shocks compress by r -> 4 (gamma=5/3), giving the universal E^(-2)
     cosmic-ray spectrum (p = 2).
  2. Weaker shocks compress less and give steeper (larger p) spectra.
  3. First-order (shock) energy gain ~ beta beats second-order (cloud) ~ beta^2 by
     a factor 1/beta.
  4. p = (r+2)/(r-1), and the power law falls off as E^(-p).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fermi_acceleration import (compression_ratio, spectral_index,  # noqa: E402
                                spectral_index_from_mach, energy_gain_per_cycle,
                                escape_probability, power_law, strong_shock_index)


def test_strong_shock_compression():
    r = compression_ratio(1e6)  # M -> infinity
    assert abs(r - 4.0) < 1e-3, f"strong-shock compression {r} not ~4"


def test_universal_e_minus_2():
    assert abs(strong_shock_index() - 2.0) < 1e-9
    assert abs(spectral_index(4.0) - 2.0) < 1e-9


def test_weaker_shock_steeper():
    p_strong = spectral_index_from_mach(10.0)
    p_weak = spectral_index_from_mach(2.0)
    assert p_weak > p_strong, "a weaker shock should give a steeper spectrum"
    assert p_strong > 2.0 or abs(p_strong - 2.0) < 0.1


def test_compression_capped_at_four():
    # no Mach number exceeds the r=4 strong-shock limit for gamma=5/3
    for M in (2, 5, 10, 100, 1e4):
        assert compression_ratio(M) < 4.0 + 1e-6


def test_first_order_beats_second():
    beta = 0.05
    g1 = energy_gain_per_cycle(beta, order=1)
    g2 = energy_gain_per_cycle(beta, order=2)
    assert g1 > g2
    assert abs(g1 / g2 - 1.0 / beta) < 1e-9


def test_index_formula():
    for r in (2.0, 3.0, 3.5, 4.0):
        assert abs(spectral_index(r) - (r + 2.0) / (r - 1.0)) < 1e-12


def test_power_law_falls_off():
    # a decade in energy drops N by 10^(-p)
    assert abs(power_law(10.0, 1.0, 2.0) - 0.01) < 1e-12
    assert power_law(100.0, 1.0, 2.0) < power_law(10.0, 1.0, 2.0)


def test_escape_probability_scales_with_speed():
    assert abs(escape_probability(2e6) / escape_probability(1e6) - 2.0) < 1e-9


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
