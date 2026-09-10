"""Binary-pulsar (Hulse-Taylor) timing tests.

Claims checked:
  1. PSR B1913+16's orbital-period decay from gravitational-wave emission is
     dP/dt ~ -2.4e-12 s/s, matching the measured value to better than 1%.
  2. The decay is negative (the orbit shrinks) and the cumulative periastron
     shift is a downward parabola -- ~40 s over 30 years, the famous plot.
  3. dP/dt strengthens with eccentricity (the (1-e^2)^{-7/2} enhancement) and
     with a shorter orbital period.
  4. The current P/|Pdot| timescale is a few hundred Myr.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulsar import (period_decay_rate, hulse_taylor_pdot,  # noqa: E402
                    cumulative_periastron_shift, merger_time,
                    HT_P_ORB, HT_E, HT_M1, HT_M2, M_SUN, YEAR)


def test_hulse_taylor_period_decay():
    pdot = hulse_taylor_pdot()
    measured = -2.423e-12
    assert abs(pdot - measured) / abs(measured) < 0.01, f"HT dP/dt {pdot} not ~{measured}"


def test_decay_is_negative():
    assert hulse_taylor_pdot() < 0.0, "orbit should shrink (GW loss)"


def test_cumulative_shift_parabola():
    # 30-year cumulative shift ~ -40 s, and it scales as t^2
    s30 = cumulative_periastron_shift(30 * YEAR)
    assert -45.0 < s30 < -30.0, f"30-yr shift {s30} not ~-40 s"
    s15 = cumulative_periastron_shift(15 * YEAR)
    assert abs(s30 / s15 - 4.0) < 1e-6, "cumulative shift should scale as t^2"


def test_eccentricity_enhances_decay():
    circ = period_decay_rate(HT_P_ORB, 0.0, HT_M1, HT_M2)
    ecc = period_decay_rate(HT_P_ORB, 0.6, HT_M1, HT_M2)
    assert abs(ecc) > abs(circ), "eccentric orbits radiate faster"


def test_shorter_period_decays_faster():
    slow = period_decay_rate(2 * HT_P_ORB, HT_E, HT_M1, HT_M2)
    fast = period_decay_rate(HT_P_ORB, HT_E, HT_M1, HT_M2)
    assert abs(fast) > abs(slow), "tighter (shorter-period) orbits decay faster"


def test_merger_timescale():
    t_myr = merger_time() / YEAR / 1e6
    assert 200.0 < t_myr < 500.0, f"P/|Pdot| {t_myr} Myr off"


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
