"""Kozai-Lidov secular-dynamics tests.

Claims checked:
  1. The conserved quantity Theta = sqrt(1-e^2) cos i holds to machine precision
     along a full cycle.
  2. Below the critical inclination (~39.2 deg) a near-circular orbit stays
     near-circular; above it the eccentricity is excited to large values.
  3. The maximum eccentricity from a near-circular start matches the analytic
     e_max = sqrt(1 - (5/3) cos^2 i0).
  4. Eccentricity and inclination oscillate out of phase: when e peaks, i dips to
     its minimum (they trade the fixed Theta).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kozai import evolve, analytic_emax, theta_of, CRITICAL_ANGLE_DEG  # noqa: E402


def test_theta_conserved():
    taus, es, idegs, Fs = evolve(0.01, 70.0, dtau=1e-3, n_steps=120000, sample_every=500)
    th = [theta_of(es[k], idegs[k]) for k in range(len(es))]
    assert max(th) - min(th) < 1e-9, f"Theta not conserved: drift {max(th)-min(th)}"


def test_below_critical_no_excitation():
    taus, es, idegs, Fs = evolve(0.01, 30.0, dtau=1e-3, n_steps=120000, sample_every=500)
    assert max(es) < 0.05, f"below critical angle e should stay small, got {max(es)}"


def test_above_critical_excites_eccentricity():
    taus, es, idegs, Fs = evolve(0.01, 70.0, dtau=1e-3, n_steps=120000, sample_every=500)
    assert max(es) > 0.5, f"above critical angle e should be excited, got {max(es)}"


def test_emax_matches_analytic():
    for i0 in (55.0, 70.0, 85.0):
        taus, es, idegs, Fs = evolve(0.01, i0, dtau=1e-3, n_steps=150000, sample_every=500)
        pred = analytic_emax(i0)
        assert abs(max(es) - pred) < 0.02, f"i0={i0}: e_max {max(es)} vs analytic {pred}"


def test_e_and_i_anticorrelate():
    taus, es, idegs, Fs = evolve(0.01, 75.0, dtau=1e-3, n_steps=150000, sample_every=300)
    kmax = max(range(len(es)), key=lambda k: es[k])
    # at maximum eccentricity, inclination is near its minimum
    assert idegs[kmax] < min(idegs) + 2.0, "inclination should dip when e peaks"


def test_critical_angle_value():
    assert abs(CRITICAL_ANGLE_DEG - 39.2314) < 1e-3


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
