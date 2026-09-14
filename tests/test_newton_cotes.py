"""Tests for Newton-Cotes: polynomial exactness to degree, weight sums, composite order, known integrals."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from newton_cotes import (  # noqa: E402
    integrate, composite, degree_of_exactness, weight_sum, error_constant, rule_names,
    trapezoid, simpson,
)
import quadrature  # noqa: E402


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
    # ---- 1. each rule is exact for polynomials up to its degree of exactness ------------
    for rule in rule_names():
        d = degree_of_exactness(rule)
        ok = True
        for deg in range(d + 1):
            val = integrate(lambda x, deg=deg: x ** deg, 0, 1, rule)
            if abs(val - 1.0 / (deg + 1)) > 1e-10:
                ok = False
                break
        check(f"{rule} exact to degree {d}", ok)

    # ---- 2. Simpson (even points) exact one degree beyond its point count ---------------
    # Simpson (3 points, would be degree 2 by count) integrates cubics exactly (degree 3)
    check("Simpson integrates cubic exactly",
          abs(integrate(lambda x: x ** 3, 0, 1, "simpson") - 0.25) < 1e-12)
    # Boole (5 points) integrates degree 5 exactly
    check("Boole integrates quintic exactly",
          abs(integrate(lambda x: x ** 5, 0, 1, "boole") - 1.0 / 6) < 1e-12)

    # ---- 3. each rule FAILS one degree past its exactness (sanity) ----------------------
    # trapezoid (degree 1) is not exact for x^2
    check("trapezoid not exact for quadratic",
          abs(integrate(lambda x: x ** 2, 0, 1, "trapezoid") - 1.0 / 3) > 1e-3)

    # ---- 4. weight sums equal interval length (npts - 1) --------------------------------
    # h * sum(w) = (b-a) means sum(w) = npts-1 for a unit-spacing interval
    from newton_cotes import _WEIGHTS
    for rule in rule_names():
        npts = len(_WEIGHTS[rule])
        check(f"{rule} weights sum = npts-1", abs(weight_sum(rule) - (npts - 1)) < 1e-12,
              f"{weight_sum(rule)} vs {npts-1}")

    # ---- 5. known integrals via composite rules -----------------------------------------
    check("composite Simpson e^x",
          abs(composite(math.exp, 0, 1, "simpson", 50) - (math.e - 1)) < 1e-9)
    check("composite Boole 1/(1+x^2)",
          abs(composite(lambda x: 1 / (1 + x * x), -1, 1, "boole", 20) - math.pi / 2) < 1e-9)
    check("composite trapezoid sin [0,pi]",
          abs(composite(math.sin, 0, math.pi, "trapezoid", 1000) - 2.0) < 1e-5)

    # ---- 6. composite Simpson converges at O(h^4), trapezoid at O(h^2) -------------------
    f = lambda x: math.exp(x)
    ref = math.e - 1
    # halving panels should reduce error ~16x for Simpson, ~4x for trapezoid
    e_s1 = abs(composite(f, 0, 1, "simpson", 5) - ref)
    e_s2 = abs(composite(f, 0, 1, "simpson", 10) - ref)
    e_t1 = abs(composite(f, 0, 1, "trapezoid", 5) - ref)
    e_t2 = abs(composite(f, 0, 1, "trapezoid", 10) - ref)
    check("Simpson error ~ /16 per halving", 10 < e_s1 / e_s2 < 25, f"ratio {e_s1/e_s2:.1f}")
    check("trapezoid error ~ /4 per halving", 3 < e_t1 / e_t2 < 5, f"ratio {e_t1/e_t2:.1f}")

    # ---- 7. Simpson far more accurate than trapezoid at equal panels --------------------
    check("Simpson beats trapezoid", e_s2 < e_t2 / 100, f"S {e_s2:.2e} T {e_t2:.2e}")

    # ---- 8. agrees with the repo's quadrature module ------------------------------------
    ref_trap = quadrature.trapezoid(f, 0, 1, 100)
    my_trap = composite(f, 0, 1, "trapezoid", 100)
    check("trapezoid == quadrature.trapezoid", abs(ref_trap - my_trap) < 1e-9,
          f"{ref_trap} vs {my_trap}")
    ref_simp = quadrature.simpson(f, 0, 1, 100)
    my_simp = composite(f, 0, 1, "simpson", 50)
    check("Simpson ~ quadrature.simpson", abs(ref_simp - my_simp) < 1e-9)

    # ---- 9. error constants exposed -----------------------------------------------------
    c, p, k = error_constant("simpson")
    check("Simpson error constant", c == "-1/90" and p == 5 and k == 4)
    c, p, k = error_constant("trapezoid")
    check("trapezoid error O(h^2)", k == 2)

    # ---- 10. convenience wrappers -------------------------------------------------------
    check("trapezoid() wrapper", abs(trapezoid(f, 0, 1, 1000) - ref) < 1e-5)
    check("simpson() wrapper", abs(simpson(f, 0, 1, 50) - ref) < 1e-9)

    # ---- 11. constant and linear exact for all rules ------------------------------------
    ok = True
    for rule in rule_names():
        if abs(integrate(lambda x: 3.0, 2, 5, rule) - 9.0) > 1e-10:  # 3*(5-2)
            ok = False
        if abs(integrate(lambda x: x, 0, 2, rule) - 2.0) > 1e-10:    # int x [0,2] = 2
            ok = False
    check("all rules exact for constant and linear", ok)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
