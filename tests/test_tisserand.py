"""Tisserand-parameter tests.

Claims checked:
  1. The formula reduces correctly: for a coplanar orbit equal to the planet's
     (a=a_p, e=0, i=0) the Tisserand parameter is exactly 3.
  2. Across a real gravity-assist flyby the semi-major axis and eccentricity
     change substantially, yet the Tisserand parameter is nearly conserved
     (before vs after, measured away from the close encounter).
  3. The Tisserand parameter classifies orbits: a tighter/rounder orbit has a
     larger T than a wide eccentric one crossing the same planet.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tisserand import tisserand, flyby  # noqa: E402


def test_matches_planet_orbit_gives_three():
    assert abs(tisserand(1.0, 0.0, 0.0, 1.0) - 3.0) < 1e-12


def _before_after(out, frac=0.2):
    k = max(1, int(len(out) * frac))
    before, after = out[:k], out[-k:]

    def avg(seg, idx):
        return sum(o[idx] for o in seg) / len(seg)
    return (avg(before, 1), avg(before, 2), avg(before, 3),
            avg(after, 1), avg(after, 2), avg(after, 3))


def test_flyby_changes_orbit_but_conserves_tisserand():
    out = flyby(a0=1.6, e0=0.4, a_planet=1.0, m_planet=1e-3,
                dt=5e-4, steps=80000, sample_every=100)
    a0, e0, T0, a1, e1, T1 = _before_after(out)
    # the orbit is substantially reshaped
    assert abs(a1 - a0) > 0.1, f"flyby should change a noticeably: {a0}->{a1}"
    # but the Tisserand parameter barely moves
    assert abs(T1 - T0) < 0.02, f"Tisserand should be near-invariant: {T0}->{T1}"


def test_tisserand_ordering():
    # a near-circular orbit close to the planet vs a wide eccentric crosser
    T_tight = tisserand(1.05, 0.05, 0.0, 1.0)
    T_wide = tisserand(3.0, 0.7, 0.0, 1.0)
    assert T_tight > T_wide, f"tighter/rounder orbit should have larger T: {T_tight} vs {T_wide}"


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
