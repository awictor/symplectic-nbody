"""Peters (1964) eccentric-inspiral tests.

The famous result: gravitational-wave emission CIRCULARIZES a binary. Both the
semi-major axis and the eccentricity shrink, with e falling faster and faster as
merger approaches, so binaries are nearly circular by the time they merge.

Claims checked:
  1. Both da/dt and de/dt are negative for e>0 (orbit shrinks AND circularizes).
  2. A circular orbit (e=0) stays circular: de/dt = 0.
  3. The Peters ODEs agree with a DIRECT 2.5PN integration on the de/da ratio
     (the analytic orbit-average matches the resolved dynamics).
  4. Over a full evolution the eccentricity monotonically decreases.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gravwave import (  # noqa: E402
    peters_dadt, peters_dedt, peters_evolve, _rk4_rel,
)

G, M, mu = 1.0, 1.0, 0.25
m1 = m2 = 0.5


def test_both_rates_negative():
    da = peters_dadt(1.0, 0.5, m1, m2, c=10.0)
    de = peters_dedt(1.0, 0.5, m1, m2, c=10.0)
    assert da < 0 and de < 0, f"orbit should shrink and circularize: da={da}, de={de}"


def test_circular_stays_circular():
    de = peters_dedt(1.0, 0.0, m1, m2, c=10.0)
    assert de == 0.0, f"e=0 must have de/dt=0, got {de}"


def test_peters_matches_direct_integration():
    c = 8.0
    a0, e0 = 1.0, 0.5

    # direct 2.5PN integration, measure elements after 12 orbits
    r_apo = a0 * (1 + e0)
    v_apo = math.sqrt(M * (2.0 / r_apo - 1.0 / a0))
    r, v = [r_apo, 0.0, 0.0], [0.0, v_apo, 0.0]
    period = 2 * math.pi * math.sqrt(a0 ** 3 / M)
    dt = period / 3000

    def elements(r, v):
        d = math.sqrt(sum(x * x for x in r))
        vv = sum(x * x for x in v)
        E = 0.5 * vv - M / d
        L = r[0] * v[1] - r[1] * v[0]
        a_ = -M / (2 * E)
        e_ = math.sqrt(max(0.0, 1 + 2 * E * L * L / (M * M)))
        return a_, e_

    for _ in range(3000 * 12):
        r, v = _rk4_rel(r, v, m1, m2, c, dt)
    a_dir, e_dir = elements(r, v)
    deda_direct = (e_dir - e0) / (a_dir - a0)

    # analytic de/da at the start
    deda_peters = peters_dedt(a0, e0, m1, m2, c) / peters_dadt(a0, e0, m1, m2, c)
    rel = abs(deda_direct - deda_peters) / abs(deda_peters)
    assert rel < 0.1, f"de/da mismatch: direct {deda_direct:.4f} vs Peters {deda_peters:.4f}"


def test_eccentricity_monotonically_decreases():
    ts, a, e = peters_evolve(1.0, 0.6, m1, m2, c=8.0, dt=0.05,
                             n_steps=30000, sample_every=500)
    for i in range(1, len(e)):
        assert e[i] <= e[i - 1] + 1e-12, f"eccentricity rose at step {i}: {e[i-1]} -> {e[i]}"
    assert e[-1] < e[0], f"eccentricity should decay: {e[0]} -> {e[-1]}"


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
