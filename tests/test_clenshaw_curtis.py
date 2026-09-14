"""Tests for Clenshaw-Curtis quadrature: weight sum, polynomial exactness, known integrals, spectral."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clenshaw_curtis import (  # noqa: E402
    nodes_weights, integrate, integrate_adaptive, weight_sum, trapezoid,
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
    # ---- 1. weights sum to the interval length ------------------------------------------
    for n, a, b in [(4, -1, 1), (8, 0, 1), (16, -2, 3), (32, 0, 10)]:
        check(f"weights sum to b-a (n={n},[{a},{b}])", abs(weight_sum(n, a, b) - (b - a)) < 1e-12)

    # ---- 2. exact for polynomials up to degree n ----------------------------------------
    # integral of x^k over [0,1] = 1/(k+1)
    ok = True
    n = 10
    for k in range(n + 1):
        val = integrate(lambda x, k=k: x ** k, 0, 1, n)
        if abs(val - 1.0 / (k + 1)) > 1e-9:
            ok = False
            check("polynomial exactness", False, f"x^{k}: {val} vs {1/(k+1)}")
            break
    if ok:
        check("exact for polynomials up to degree n (x^0..x^10 on [0,1])", True)

    # ---- 3. known integrals -------------------------------------------------------------
    check("int e^x [0,1] = e-1",
          abs(integrate(math.exp, 0, 1, 32) - (math.e - 1)) < 1e-12)
    check("int 1/(1+x^2) [-1,1] = pi/2",
          abs(integrate(lambda x: 1 / (1 + x * x), -1, 1, 48) - math.pi / 2) < 1e-10)
    check("int cos [0, 2pi] = 0",
          abs(integrate(math.cos, 0, 2 * math.pi, 32)) < 1e-10)
    check("int sin [0, pi] = 2",
          abs(integrate(math.sin, 0, math.pi, 32) - 2.0) < 1e-10)
    check("int sqrt(x) [0,1] = 2/3",
          abs(integrate(math.sqrt, 0, 1, 128) - 2.0 / 3.0) < 1e-4)  # endpoint singularity in deriv

    # ---- 4. spectral convergence beats the trapezoidal rule -----------------------------
    # smooth integrand: e^{cos x} on [0, 2pi]; compare error at matched node count
    f = lambda x: math.exp(math.cos(x))
    a, b = 0.0, 2.0 * math.pi
    # high-accuracy reference
    ref = integrate(f, a, b, 256)
    cc_err = abs(integrate(f, a, b, 16) - ref)
    tr_err = abs(trapezoid(f, a, b, 16) - ref)
    # (for a periodic function the trapezoidal rule is itself spectral, so use a non-periodic one)
    g = lambda x: 1.0 / (2 + x)
    refg = integrate(g, 0, 1, 256)
    cc_errg = abs(integrate(g, 0, 1, 8) - refg)
    tr_errg = abs(trapezoid(g, 0, 1, 8) - refg)
    check("Clenshaw-Curtis beats trapezoid on smooth integrand", cc_errg < tr_errg,
          f"CC {cc_errg:.2e} vs trap {tr_errg:.2e}")

    # ---- 5. error decreases as n grows (spectral) ---------------------------------------
    f = lambda x: 1.0 / (1.0 + 16.0 * x * x)
    ref = integrate(f, -1, 1, 512)
    errs = [abs(integrate(f, -1, 1, n) - ref) for n in [8, 16, 32, 64]]
    check("error decreases with n", all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)),
          f"{[f'{e:.1e}' for e in errs]}")

    # ---- 6. adaptive integration converges ----------------------------------------------
    val, n = integrate_adaptive(math.exp, 0, 1, tol=1e-12)
    check("adaptive int e^x", abs(val - (math.e - 1)) < 1e-11, f"{val} in n={n}")

    # ---- 7. nodes for n and 2n nest -----------------------------------------------------
    xs1, _ = nodes_weights(8, -1, 1)
    xs2, _ = nodes_weights(16, -1, 1)
    # every node of the n=8 rule appears in the n=16 rule
    s2 = set(round(x, 10) for x in xs2)
    nested = all(round(x, 10) in s2 for x in xs1)
    check("nodes nest (n=8 subset of n=16)", nested)

    # ---- 8. node count and endpoints ----------------------------------------------------
    xs, ws = nodes_weights(10, 2, 5)
    check("n+1 nodes", len(xs) == 11 and len(ws) == 11)
    check("endpoints are interval ends",
          abs(min(xs) - 2) < 1e-12 and abs(max(xs) - 5) < 1e-12)

    # ---- 9. constant function -----------------------------------------------------------
    check("int of 1 over [3,7] = 4", abs(integrate(lambda x: 1.0, 3, 7, 8) - 4.0) < 1e-12)

    # ---- 10. symmetric integrand --------------------------------------------------------
    # int x^2 over [-1,1] = 2/3
    check("int x^2 [-1,1] = 2/3", abs(integrate(lambda x: x * x, -1, 1, 4) - 2.0 / 3.0) < 1e-12)

    # ---- 11. validation -----------------------------------------------------------------
    try:
        nodes_weights(0)
        check("n=0 raises", False)
    except ValueError:
        check("n=0 raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
