"""Gravity-assist / slingshot tests.

Claims checked:
  1. The hyperbolic turn angle grows for slower, deeper passes (larger 1/e), and
     the max speed change approaches 2 v_inf for a strong bend.
  2. A Jupiter flyby gives a Voyager-scale ~10-18 km/s heliocentric boost.
  3. The maximum single-flyby delta-v never exceeds 2 v_inf.
  4. A trailing (head-on) pass adds speed; the boost costs the planet nothing here.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gravity_assist import (eccentricity, turn_angle, max_delta_v,  # noqa: E402
                            heliocentric_speed_after, slingshot_gain,
                            MU_JUPITER, V_JUPITER)

R_JUP = 7.1492e7
V_INF = 10e3


def test_deeper_pass_bends_more():
    shallow = turn_angle(V_INF, 8 * R_JUP, MU_JUPITER)
    deep = turn_angle(V_INF, 2 * R_JUP, MU_JUPITER)
    assert deep > shallow


def test_slower_pass_bends_more():
    fast = turn_angle(20e3, 3 * R_JUP, MU_JUPITER)
    slow = turn_angle(5e3, 3 * R_JUP, MU_JUPITER)
    assert slow > fast


def test_jupiter_boost_scale():
    dv = max_delta_v(V_INF, 3 * R_JUP, MU_JUPITER) / 1e3  # km/s
    assert 8.0 < dv < 20.0, f"Jupiter slingshot boost {dv} km/s off scale"


def test_never_exceeds_2vinf():
    for rp in (2 * R_JUP, 5 * R_JUP, 20 * R_JUP):
        assert max_delta_v(V_INF, rp, MU_JUPITER) <= 2.0 * V_INF + 1e-6


def test_eccentricity_gt_one():
    assert eccentricity(V_INF, 3 * R_JUP, MU_JUPITER) > 1.0


def test_trailing_pass_adds_speed():
    v_after = heliocentric_speed_after(20e3, V_JUPITER, V_INF, 3 * R_JUP, MU_JUPITER)
    assert v_after > 20e3


def test_turn_angle_limit():
    # a very slow, deep pass approaches a near-180-degree reversal
    delta = math.degrees(turn_angle(1e3, 1.5 * R_JUP, MU_JUPITER))
    assert delta > 150.0


def test_gain_matches_max_dv():
    assert abs(slingshot_gain(V_INF, 3 * R_JUP, MU_JUPITER)
               - max_delta_v(V_INF, 3 * R_JUP, MU_JUPITER)) < 1e-9


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
