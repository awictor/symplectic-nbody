"""Two-body relaxation / evaporation tests.

Claims checked:
  1. A globular cluster (N ~ 1e5) relaxes in ~1 Gyr -- collisional within a Hubble
     time; a galaxy (N ~ 1e11) has t_relax >> Hubble time and is collisionless.
  2. t_relax ~ N / ln N * t_cross, so it grows almost linearly with N.
  3. The evaporation time is a large multiple of the relaxation time.
  4. Denser/faster systems (shorter crossing time) relax faster.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relaxation_time import (crossing_time, relaxation_time,  # noqa: E402
                             relaxation_time_from_params, is_collisionless,
                             evaporation_time, PC, MYR, GYR, HUBBLE_TIME,
                             EVAP_FRACTION)


def test_globular_relaxes_within_hubble():
    tc = crossing_time(10 * PC, 10e3)
    tr = relaxation_time(int(1e5), tc) / GYR
    assert 0.3 < tr < 3.0, f"globular t_relax {tr} Gyr off scale"
    assert not is_collisionless(int(1e5), tc)


def test_galaxy_is_collisionless():
    tc = crossing_time(15000 * PC, 200e3)
    assert is_collisionless(int(1e11), tc)
    assert relaxation_time(int(1e11), tc) > 1000.0 * HUBBLE_TIME


def test_grows_with_N():
    tc = 1.0 * MYR
    # roughly linear in N (N/ln N), so 100x N gives well over 50x t_relax
    ratio = relaxation_time(int(1e6), tc) / relaxation_time(int(1e4), tc)
    assert ratio > 50.0


def test_evaporation_is_large_multiple():
    tc = crossing_time(10 * PC, 10e3)
    tr = relaxation_time(int(1e5), tc)
    te = evaporation_time(int(1e5), tc)
    assert te > 50.0 * tr
    assert abs(te - tr / EVAP_FRACTION) < 1.0


def test_faster_crossing_relaxes_faster():
    slow = relaxation_time_from_params(int(1e5), 10 * PC, 5e3)
    fast = relaxation_time_from_params(int(1e5), 10 * PC, 20e3)
    assert fast < slow


def test_crossing_time_definition():
    assert abs(crossing_time(10 * PC, 10e3) - 10 * PC / 10e3) < 1.0


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
