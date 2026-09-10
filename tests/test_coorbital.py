"""Coorbital-motion tests: tadpole vs horseshoe orbits in the CR3BP.

Claims checked:
  1. A small nudge from L4 gives a TADPOLE orbit: the particle librates on one
     side of the primary-secondary line (angular range < 180 deg) and stays
     bounded near L4.
  2. Starting on the corotation circle opposite the secondary (near L3) gives a
     HORSESHOE orbit: the particle sweeps across the far side (angular range
     > 180 deg) but never reaches the secondary (stays within the unit circle).
  3. classify() labels them correctly.
  4. The angular-range measure is orientation-consistent (unwraps the wrap).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cr3bp import CR3BP  # noqa: E402
from coorbital import (coorbital_trajectory, angular_range, classify,  # noqa: E402
                       start_near_L4, start_on_corotation)


def test_tadpole_orbit():
    m = CR3BP(0.001)
    x0, y0 = start_near_L4(m, offset=0.008)
    xs, ys, angs = coorbital_trajectory(m, x0, y0, dt=0.005, steps=150000,
                                        sample_every=100)
    rng = angular_range(angs)
    assert rng < 180.0, f"tadpole should stay on one side, range {rng}"
    assert classify(angs) == "tadpole"


def test_horseshoe_orbit():
    m = CR3BP(0.001)
    x0, y0 = start_on_corotation(180.0)
    xs, ys, angs = coorbital_trajectory(m, x0, y0, dt=0.005, steps=300000,
                                        sample_every=100)
    rng = angular_range(angs)
    assert rng > 180.0, f"horseshoe should sweep the far side, range {rng}"
    assert classify(angs) == "horseshoe"
    # never reaches the secondary at (1-mu, 0): stays within the unit circle
    assert max(xs) < 1.05, f"horseshoe should turn back before the secondary, max x {max(xs)}"


def test_angular_range_unwrap():
    # a sequence that crosses the 0/2pi seam should report a small range
    angs = [math.radians(a % 360) for a in (350, 355, 0, 5, 10)]
    assert angular_range(angs) < 30.0, "angular_range must unwrap the 0/2pi seam"


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
