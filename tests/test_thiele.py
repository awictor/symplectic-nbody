"""Tests for Thiele rational interpolation: passes nodes, recovers rationals, poles, beats polynomial."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thiele import (  # noqa: E402
    ThieleInterpolant, thiele_fit, reciprocal_differences, thiele_eval,
    lagrange_eval, runge,
)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    # ---- 1. interpolant passes through every data node ----------------------------------
    ok = True
    for f in [lambda x: (x + 1) / (x * x + 1), lambda x: math.sin(x), lambda x: x ** 3 - 2 * x]:
        xs = [-2, -1, 0, 1, 2, 3]
        t = thiele_fit(f, xs)
        for x in xs:
            if abs(t(x) - f(x)) > 1e-7:
                ok = False
                check("passes nodes", False, f"x={x}: {t(x)} vs {f(x)}")
                break
        if not ok:
            break
    if ok:
        check("interpolant passes through all data nodes", True)

    # ---- 2. recovers a known rational function off-node to machine precision ------------
    r = lambda x: (x + 1) / (x * x + 1)
    xs = [-2, -1, 0, 1, 2, 3]
    t = thiele_fit(r, xs)
    off = [-1.5, -0.5, 0.5, 1.5, 2.5]
    max_err = max(abs(t(x) - r(x)) for x in off)
    check("recovers rational (x+1)/(x^2+1)", max_err < 1e-9, f"max err {max_err:.2e}")

    # ---- 3. recovers a low-degree polynomial (degenerate rational) exactly --------------
    p = lambda x: 2 * x * x - 3 * x + 1
    xs = [0, 1, 2, 3, 4]
    t = thiele_fit(p, xs)
    check("recovers polynomial 2x^2-3x+1", all(abs(t(x) - p(x)) < 1e-9 for x in [0.5, 1.5, 2.5, 3.5]))

    # ---- 4. reproduces a function with a pole ------------------------------------------
    pf = lambda x: 1.0 / (x - 0.5)
    xs = [-2, -1, 0, 1, 2, 3]
    t = thiele_fit(pf, xs)
    off = [0.7, 1.3, 2.2, -1.5]
    max_err = max(abs(t(x) - pf(x)) for x in off)
    check("reproduces pole 1/(x-0.5)", max_err < 1e-8, f"max err {max_err:.2e}")

    # ---- 5. Runge function: Thiele beats polynomial dramatically ------------------------
    # 1/(1+25x^2) is rational, so Thiele nails it while Lagrange (equispaced) diverges
    xs = [i * 0.2 - 1 for i in range(11)]
    ys = [runge(x) for x in xs]
    t = ThieleInterpolant(xs, ys)
    test_pts = [i * 0.05 - 1 for i in range(41)]
    thiele_err = max(abs(t(x) - runge(x)) for x in test_pts)
    lagrange_err = max(abs(lagrange_eval(xs, ys, x) - runge(x)) for x in test_pts)
    check("Thiele nails the Runge function", thiele_err < 1e-6, f"{thiele_err:.2e}")
    check("Thiele << Lagrange on Runge", thiele_err < lagrange_err / 100,
          f"thiele {thiele_err:.2e} vs lagrange {lagrange_err:.3f}")

    # ---- 6. matches the polynomial when given MORE nodes than its degree ----------------
    # With exactly deg+1 nodes Thiele returns a [1/1]-type rational agreeing at the nodes but not
    # off-node; with extra nodes the reciprocal differences blow up past the degree and the fraction
    # truncates to the exact polynomial.
    p = lambda x: x ** 2 + 1
    xs = [0, 1, 2, 3, 4]
    t = thiele_fit(p, xs)
    for x in [0.3, 1.7, 3.4]:
        check(f"poly recovered with extra nodes at {x}", abs(t(x) - p(x)) < 1e-9,
              f"{t(x)} vs {p(x)}")

    # ---- 7. arctangent (near-rational) -------------------------------------------------
    xs = [-2, -1, -0.5, 0, 0.5, 1, 2]
    t = thiele_fit(math.atan, xs)
    err = max(abs(t(x) - math.atan(x)) for x in [-1.5, -0.25, 0.25, 1.5])
    check("arctan interpolated well", err < 0.05, f"{err:.4f}")

    # ---- 8. two-point (linear) case -----------------------------------------------------
    t = ThieleInterpolant([0, 1], [1, 3])
    check("two points -> linear through them", abs(t(0.5) - 2.0) < 1e-9, f"{t(0.5)}")

    # ---- 9. reciprocal_differences first coefficient is y_0 -----------------------------
    coeffs = reciprocal_differences([1, 2, 3], [10, 20, 30])
    check("first coefficient is y_0", coeffs[0] == 10)

    # ---- 10. constant function ----------------------------------------------------------
    t = thiele_fit(lambda x: 5.0, [0, 1, 2])
    check("constant function", abs(t(0.7) - 5.0) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
