"""Post-Newtonian precession tests: reproduce the Mercury 43"/century result.

Claims checked:
  1. The analytic 1PN formula gives Mercury ~43 arcsec/century (the famous
     value that vindicated general relativity).
  2. Numerically integrating the 1PN equations of motion reproduces the closed-
     form perihelion advance 6*pi*GM/(c^2 a (1-e^2)) -- checked in a regime where
     the effect is large enough to measure cleanly but 1PN is still valid.
  3. Newtonian orbits do NOT precess (perihelion advance ~ 0), isolating the
     precession as a purely relativistic effect.
  4. Precession scales as 1/c^2.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativity import (  # noqa: E402
    analytic_precession_per_orbit, precession_per_orbit,
    mercury_precession_arcsec_per_century, C_LIGHT, GM_SUN,
)

MERCURY_A, MERCURY_E = 0.387098, 0.205630


def test_mercury_famous_43_arcsec():
    val = mercury_precession_arcsec_per_century()
    assert abs(val - 43.0) < 1.0, f"Mercury precession {val} not ~43 arcsec/century"


def test_numeric_matches_analytic():
    # amplify GR by shrinking c 300x: effect is measurable, 1PN still valid
    c = C_LIGHT / 300.0
    num = precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, c,
                               orbits=20, steps_per_orbit=12000)
    ana = analytic_precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, c)
    rel = abs(num - ana) / ana
    assert rel < 0.02, f"numeric precession off by {rel:.3%} (num {num}, ana {ana})"


def test_precession_is_prograde():
    c = C_LIGHT / 300.0
    num = precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, c,
                               orbits=20, steps_per_orbit=12000)
    assert num > 0, f"GR perihelion advance should be prograde, got {num}"


def test_newtonian_does_not_precess():
    # c -> infinity kills the 1PN term; use a huge c so precession ~ 0
    c = C_LIGHT * 1e6
    num = precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, c,
                               orbits=20, steps_per_orbit=12000)
    ana = analytic_precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, C_LIGHT / 300.0)
    assert abs(num) < 0.01 * ana, f"Newtonian orbit should not precess, got {num}"


def test_scales_as_inverse_c_squared():
    a1 = analytic_precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, C_LIGHT / 300.0)
    a2 = analytic_precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, C_LIGHT / 600.0)
    # halving c should quadruple the precession
    assert abs(a2 / a1 - 4.0) < 1e-6, f"precession not ~1/c^2: ratio {a2/a1}"


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
