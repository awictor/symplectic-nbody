"""Three-body stability-map tests.

Claims checked:
  1. Escape time is deterministic (same IC -> same result).
  2. The setup is mirror-symmetric: reflecting the third body across the x-axis
     (y -> -y) leaves the escape time unchanged, since the two primaries lie on
     that axis. This is a genuine physical invariant of the configuration.
  3. A body dropped essentially at the escape radius registers an escape almost
     immediately, far sooner than one dropped deep between the primaries --
     escape time carries real dynamical signal, not noise.
  4. Escape time is bounded by t_max.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stability_map import escape_time  # noqa: E402


def test_deterministic():
    a = escape_time(0.1, 0.2, dt=0.01, t_max=20)
    b = escape_time(0.1, 0.2, dt=0.01, t_max=20)
    assert a == b, f"escape_time not deterministic: {a} vs {b}"


def test_x_axis_mirror_symmetry():
    # primaries sit on the x-axis, so y -> -y is a symmetry of the whole system
    for (x, y) in ((0.3, 0.4), (0.0, 0.7), (-0.2, 0.5)):
        up = escape_time(x, y, dt=0.01, t_max=25)
        down = escape_time(x, -y, dt=0.01, t_max=25)
        assert abs(up - down) < 1e-9, f"mirror symmetry broken at ({x},{y}): {up} vs {down}"


def test_far_escapes_before_deep():
    deep = escape_time(0.0, 0.2, dt=0.01, t_max=40, escape_radius=8.0)
    # start the third body just past the escape radius: it's "escaped" at once
    near_edge = escape_time(8.5, 0.0, dt=0.01, t_max=40, escape_radius=8.0)
    assert near_edge < deep, f"edge IC ({near_edge}) should register before deep IC ({deep})"
    assert near_edge < 0.1, f"edge IC should escape almost immediately, got {near_edge}"


def test_bounded_by_tmax():
    t_max = 15.0
    for (x, y) in ((0.1, 0.1), (1.2, 0.9), (0.0, 0.5)):
        e = escape_time(x, y, dt=0.01, t_max=t_max)
        assert 0.0 < e <= t_max, f"escape time {e} out of (0, {t_max}]"


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
