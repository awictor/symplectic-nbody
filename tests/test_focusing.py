"""Gravitational-focusing tests.

Claims checked:
  1. The collision cross-section is pi R^2 (1 + v_esc^2/v_inf^2); at high v_inf it
     reduces to the geometric pi R^2.
  2. Slow encounters (v_inf << v_esc) give large focusing enhancement -- runaway
     growth, when the Safronov number Theta = v_esc^2/(2 v_inf^2) exceeds 1.
  3. Theta scales as 1/v_inf^2, and the runaway/ordered boundary is Theta = 1
     (v_inf = v_esc/sqrt(2)).
  4. Bigger bodies (higher escape speed) focus more strongly.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from focusing import (escape_speed, geometric_cross_section, focusing_factor,  # noqa: E402
                      collision_cross_section, safronov_number, is_runaway)

R = 1e5
RHO = 3000.0
M = 4.0 / 3.0 * math.pi * R ** 3 * RHO


def test_geometric_limit_at_high_speed():
    assert abs(focusing_factor(M, R, 1e6) - 1.0) < 1e-3, "high v_inf -> geometric"
    assert abs(collision_cross_section(M, R, 1e6) - geometric_cross_section(R))\
        / geometric_cross_section(R) < 1e-3


def test_runaway_at_low_speed():
    assert is_runaway(M, R, 1.0), "slow encounters should be runaway"
    assert focusing_factor(M, R, 1.0) > 1000.0, "large enhancement at low v"


def test_runaway_boundary():
    # Theta = 1 when v_inf = v_esc / sqrt(2)
    ve = escape_speed(M, R)
    v_boundary = ve / math.sqrt(2.0)
    assert abs(safronov_number(M, R, v_boundary) - 1.0) < 1e-9
    assert not is_runaway(M, R, 2.0 * v_boundary)
    assert is_runaway(M, R, 0.5 * v_boundary)


def test_safronov_inverse_v_squared():
    r = safronov_number(M, R, 10.0) / safronov_number(M, R, 20.0)
    assert abs(r - 4.0) < 1e-6, "Theta ~ 1/v_inf^2"


def test_bigger_bodies_focus_more():
    # a more massive (larger) body has higher escape speed -> more focusing
    M2, R2 = 8 * M, 2 * R
    assert focusing_factor(M2, R2, 50.0) > focusing_factor(M, R, 50.0)


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
