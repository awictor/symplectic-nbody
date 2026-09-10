"""Oberth-effect tests.

Claims checked:
  1. The same burn dv gains more kinetic energy at higher speed
     (dE = v dv + dv^2/2), so the energy gain rises with v.
  2. On a fixed eccentric orbit, burning at periapsis (fast) gains far more
     energy -- and more hyperbolic excess speed -- than the same burn at
     apoapsis (slow): the Oberth advantage.
  3. A burn that escapes from periapsis may leave the ship still bound if made
     at apoapsis.
  4. Vis-viva speeds are correct: periapsis speed exceeds apoapsis speed on an
     eccentric orbit.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oberth import (energy_gain, speed_at_radius, v_infinity_after_burn,  # noqa: E402
                    oberth_advantage, MU_EARTH, R_EARTH)


def test_energy_gain_rises_with_speed():
    assert energy_gain(8000, 100) > energy_gain(1000, 100)
    # the linear v dv term dominates: ratio ~ v ratio for small dv
    assert energy_gain(8000, 1) / energy_gain(1000, 1) > 7.0


def test_periapsis_burn_beats_apoapsis():
    rp, ra = R_EARTH + 300e3, R_EARTH + 35786e3
    a = 0.5 * (rp + ra)
    adv = oberth_advantage(MU_EARTH, rp, ra, 1000.0, a)
    assert adv > 3.0, f"periapsis burn should gain much more energy, ratio {adv}"


def test_same_burn_escapes_only_from_periapsis():
    rp, ra = R_EARTH + 300e3, R_EARTH + 35786e3
    a = 0.5 * (rp + ra)
    v_peri = speed_at_radius(MU_EARTH, rp, a)
    v_apo = speed_at_radius(MU_EARTH, ra, a)
    dv = 1500.0
    vinf_peri = v_infinity_after_burn(MU_EARTH, rp, v_peri, dv)
    vinf_apo = v_infinity_after_burn(MU_EARTH, ra, v_apo, dv)
    assert vinf_peri > 0.0, "periapsis burn should reach escape"
    assert vinf_apo == 0.0, "same burn at apoapsis should leave the ship bound"


def test_vis_viva_periapsis_faster():
    rp, ra = R_EARTH + 300e3, R_EARTH + 35786e3
    a = 0.5 * (rp + ra)
    assert speed_at_radius(MU_EARTH, rp, a) > speed_at_radius(MU_EARTH, ra, a)


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
