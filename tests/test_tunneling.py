"""Quantum-tunneling tests.

Claims checked:
  1. Transmission falls exponentially with barrier width and with sqrt(V-E).
  2. The exact rectangular-barrier T reduces to the thick-limit exp(-2 kappa L) for a
     wide barrier, and is 1 when E >= V.
  3. WKB on a rectangular barrier matches the analytic thick-limit result.
  4. STM tunneling current drops ~an order of magnitude per 0.1 nm of gap.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tunneling import (decay_constant, transmission_thick, transmission_exact,  # noqa: E402
                       wkb_transmission, stm_current_ratio, EV, M_E)


def test_exponential_in_width():
    t1 = transmission_thick(4 * EV, 0.5e-9)
    t2 = transmission_thick(4 * EV, 1.0e-9)
    # doubling the width squares the (small) transmission
    assert abs(t2 / t1 ** 2 - 1.0) < 1e-6


def test_higher_barrier_lower_T():
    assert transmission_thick(8 * EV, 0.5e-9) < transmission_thick(2 * EV, 0.5e-9)


def test_exact_reduces_to_thick():
    E, V, L = 1 * EV, 5 * EV, 1.0e-9   # wide barrier
    exact = transmission_exact(E, V, L)
    thick = transmission_thick(V - E, L)
    # same order of magnitude (thick limit drops the prefactor)
    assert 0.1 < exact / thick < 10.0


def test_transparent_above_barrier():
    assert transmission_exact(6 * EV, 5 * EV, 1e-9) == 1.0


def test_wkb_matches_rectangular():
    V = 4 * EV
    T_wkb = wkb_transmission(lambda x: V, 0.0, 0.5e-9, 0.0)
    T_thick = transmission_thick(V, 0.5e-9)
    assert abs(T_wkb / T_thick - 1.0) < 1e-6


def test_stm_sensitivity():
    ratio = stm_current_ratio(0.0, 1e-10)   # +1 Angstrom
    assert ratio < 0.2, "STM current should drop ~order of magnitude per Angstrom"


def test_stm_ratio_one_at_no_change():
    assert abs(stm_current_ratio(1e-10, 1e-10) - 1.0) < 1e-12


def test_decay_constant_scaling():
    # kappa ~ sqrt(V-E): quadruple the deficit doubles kappa
    assert abs(decay_constant(4 * EV) / decay_constant(1 * EV) - 2.0) < 1e-9


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
