"""Mean-motion-resonance tests.

Claims checked:
  1. Placing the outer planet at the 2:1 spacing a_out = a_in * 2^{2/3} gives a
     measured period ratio ~ 2 (Kepler's third law in reverse).
  2. In resonance the resonant argument phi LIBRATES (stays in a bounded range
     well under 2*pi); off resonance it CIRCULATES (covers the full 2*pi).
  3. Kepler spacing is consistent: a_out/a_in = (p/q)^{2/3} reproduces the ratio.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resonance import (two_planet_system, period_ratio,  # noqa: E402
                       resonant_argument_series, libration_amplitude)


def test_2to1_spacing_gives_period_ratio_2():
    a_res = 1.0 * 2 ** (2.0 / 3.0)
    s = two_planet_system(1.0, a_res, m_inner=1e-3, m_outer=1e-3, e_inner=0.05)
    pr = period_ratio(s, dt=0.002, steps=3000)
    assert abs(pr - 2.0) < 0.05, f"2:1 spacing should give period ratio ~2, got {pr}"


def test_resonant_argument_librates():
    a_res = 1.0 * 2 ** (2.0 / 3.0)
    s = two_planet_system(1.0, a_res, m_inner=2e-3, m_outer=2e-3, e_inner=0.05)
    _ts, phis = resonant_argument_series(s, 2, 1, dt=0.002, steps=40000, sample_every=50)
    amp = libration_amplitude(phis)
    assert amp < 0.9 * 2 * math.pi, f"resonant phi should librate (bounded), range {amp}"


def test_offresonance_circulates():
    s = two_planet_system(1.0, 1.8, m_inner=2e-3, m_outer=2e-3, e_inner=0.05)
    _ts, phis = resonant_argument_series(s, 2, 1, dt=0.002, steps=40000, sample_every=50)
    amp = libration_amplitude(phis)
    assert amp > 0.95 * 2 * math.pi, f"off-resonance phi should circulate, range {amp}"


def test_kepler_spacing_consistency():
    # a 3:2 resonance should sit at a_out/a_in = (3/2)^{2/3}
    a_res = 1.0 * (3.0 / 2.0) ** (2.0 / 3.0)
    s = two_planet_system(1.0, a_res, m_inner=1e-3, m_outer=1e-3, e_inner=0.02)
    pr = period_ratio(s, dt=0.002, steps=3000)
    assert abs(pr - 1.5) < 0.05, f"3:2 spacing should give period ratio ~1.5, got {pr}"


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
