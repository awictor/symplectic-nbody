"""Lyapunov-exponent tests: distinguish regular from chaotic motion.

The largest Lyapunov exponent lambda measures how fast nearby trajectories
diverge. lambda > 0 means chaos (exponential separation); regular motion has
lambda -> 0 (only power-law/linear separation).

Claims checked:
  1. A chaotic system (Burrau's pythagorean 3-body) has a clearly positive
     estimate, an order of magnitude above a regular system measured the same way.
  2. Regular motion's estimate DECAYS toward zero as the measurement time grows
     (the ~1/T signature of linear, non-exponential separation), while the
     chaotic estimate stays large.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lyapunov import largest_lyapunov  # noqa: E402
from systems import two_body_circular, pythagorean  # noqa: E402


def test_chaotic_exceeds_regular():
    reg = largest_lyapunov(two_body_circular(), dt=0.002, total_time=200)
    cha = largest_lyapunov(pythagorean(), dt=0.0005, total_time=100)
    assert cha > 5 * reg, f"chaotic ({cha}) should dwarf regular ({reg})"
    assert cha > 0.1, f"pythagorean should be clearly chaotic, got {cha}"


def test_regular_decays_with_time():
    # linear separation => estimate falls roughly like 1/T
    short = largest_lyapunov(two_body_circular(), dt=0.002, total_time=100)
    long = largest_lyapunov(two_body_circular(), dt=0.002, total_time=400)
    assert long < 0.7 * short, (
        f"regular Lyapunov estimate should decay with T: "
        f"T=100 -> {short}, T=400 -> {long}")


def test_chaotic_stays_positive():
    a = largest_lyapunov(pythagorean(), dt=0.0005, total_time=60)
    b = largest_lyapunov(pythagorean(), dt=0.0005, total_time=120)
    # both well above the regular floor (~0.02-0.06 at these times)
    assert a > 0.15 and b > 0.15, f"chaotic estimate collapsed: {a}, {b}"


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
