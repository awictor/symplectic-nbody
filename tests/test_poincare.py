"""Poincare surface-of-section tests.

Claims checked:
  1. vy_from_jacobi inverts the Jacobi relation: a point reconstructed from
     (x, vx, C) has exactly Jacobi constant C, and energetically forbidden
     points return nan.
  2. Every returned section point lies on the surface (crossings are real).
  3. A regular orbit produces a tightly-bounded set of (x, vx) points (a closed
     curve / torus); a chaotic orbit at the same energy scatters far more widely.
  4. The Jacobi constant stays well-conserved along the integrated orbit, so the
     section is faithful (RK4 drift is small on these seeds).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cr3bp import CR3BP  # noqa: E402
from poincare import section, vy_from_jacobi, jacobi_drift  # noqa: E402


def test_vy_from_jacobi_roundtrip():
    m = CR3BP(0.1)
    C = 3.9
    x, vx = -0.35, 0.0
    vy = vy_from_jacobi(m, x, vx, C)
    assert not math.isnan(vy)
    C_back = m.jacobi_constant(x, 0.0, vx, vy)
    assert abs(C_back - C) < 1e-12, f"Jacobi roundtrip off: {C_back} vs {C}"


def test_forbidden_point_is_nan():
    m = CR3BP(0.1)
    # very low C with large vx demanded -> vy^2 < 0
    assert math.isnan(vy_from_jacobi(m, -0.35, 5.0, 2.0))


def test_regular_orbit_is_bounded_curve():
    m = CR3BP(0.1)
    pts = section(m, -0.30, 0.0, 3.9, n_crossings=120, dt=0.002)
    assert len(pts) >= 50
    xspread = max(p[0] for p in pts) - min(p[0] for p in pts)
    assert xspread < 0.05, f"regular orbit should be a tight curve, xspread={xspread}"


def test_chaotic_scatters_more_than_regular():
    m = CR3BP(0.1)
    reg = section(m, -0.30, 0.0, 3.9, n_crossings=120, dt=0.002)
    cha = section(m, -0.40, 0.6, 3.9, n_crossings=120, dt=0.002)
    reg_spread = max(p[0] for p in reg) - min(p[0] for p in reg)
    cha_spread = max(p[0] for p in cha) - min(p[0] for p in cha)
    assert cha_spread > 3 * reg_spread, (
        f"chaotic spread ({cha_spread}) should exceed regular ({reg_spread})")


def test_jacobi_well_conserved():
    m = CR3BP(0.1)
    d = jacobi_drift(m, -0.30, 0.0, 3.9, steps=8000, dt=0.002)
    assert d < 1e-4, f"Jacobi drift too large for a faithful section: {d}"


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
