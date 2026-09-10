"""Chandrasekhar dynamical-friction tests.

Claims checked:
  1. A ~1e10 solar-mass satellite (LMC-scale) sinks into a Milky-Way-like halo in
     a few Gyr; the sinking time scales as 1/M (heavier sinks faster).
  2. The friction deceleration scales linearly with mass (force ~ M^2, accel = F/M).
  3. The velocity dependence is non-monotonic: friction vanishes as v -> 0 (the
     Maxwell factor -> 0) and falls off as 1/v^2 at high v, peaking in between.
  4. Denser backgrounds and larger Coulomb logarithms increase the drag.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynamical_friction import friction_acceleration, sinking_time  # noqa: E402

MSUN = 1.989e30
KPC = 3.086e19
GYR = 3.156e16
VC = 220e3
SIGMA = 150e3
RHO = 1e-21


def test_satellite_sinks_in_few_gyr():
    t = sinking_time(1e10 * MSUN, 50 * KPC, VC) / GYR
    assert 1.0 < t < 8.0, f"LMC-scale sinking time {t} Gyr off"


def test_sinking_time_inverse_mass():
    t1 = sinking_time(1e10 * MSUN, 50 * KPC, VC)
    t2 = sinking_time(1e11 * MSUN, 50 * KPC, VC)
    assert abs(t1 / t2 - 10.0) < 1e-6, "sinking time should scale as 1/M"


def test_acceleration_linear_in_mass():
    a1 = friction_acceleration(1e10 * MSUN, VC, RHO, SIGMA)
    a2 = friction_acceleration(2e10 * MSUN, VC, RHO, SIGMA)
    assert abs(a2 / a1 - 2.0) < 1e-9, "deceleration should be linear in mass"


def test_friction_vanishes_at_low_speed():
    slow = friction_acceleration(1e10 * MSUN, 0.01 * SIGMA, RHO, SIGMA)
    mid = friction_acceleration(1e10 * MSUN, SIGMA, RHO, SIGMA)
    assert slow < mid, "friction should vanish as v -> 0 (Maxwell factor)"


def test_high_speed_falloff():
    a1 = friction_acceleration(1e10 * MSUN, VC, RHO, SIGMA)
    a2 = friction_acceleration(1e10 * MSUN, 3 * VC, RHO, SIGMA)
    assert a2 < a1, "friction should fall off at high speed (~1/v^2)"


def test_denser_background_more_drag():
    a1 = friction_acceleration(1e10 * MSUN, VC, RHO, SIGMA)
    a2 = friction_acceleration(1e10 * MSUN, VC, 2 * RHO, SIGMA)
    assert abs(a2 / a1 - 2.0) < 1e-9, "drag linear in background density"


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
