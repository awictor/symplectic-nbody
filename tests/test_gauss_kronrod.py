"""Validate Gauss-Kronrod: exact polynomial degree, error estimate, adaptive resolution vs Romberg/Simpson."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import gauss_kronrod as gk
import quadrature


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Gauss-Kronrod tests")

    # --- the embedded 7-point Gauss rule is exact to degree 13 ---
    # integral of x^p on [-1,1] = 2/(p+1) for even p, 0 for odd p
    ok_gauss = True
    for p in range(0, 14):
        k, g, e = gk.gauss_kronrod_15(lambda x, p=p: x ** p, -1.0, 1.0)
        exact = 2.0 / (p + 1) if p % 2 == 0 else 0.0
        if abs(g - exact) > 1e-11:
            ok_gauss = False
    check("Gauss rule exact for degree <= 13", ok_gauss)

    # --- the 15-point Kronrod rule is exact to degree 3n+1 = 22 ---
    ok_kron = True
    for p in range(0, 23):
        k, g, e = gk.gauss_kronrod_15(lambda x, p=p: x ** p, -1.0, 1.0)
        exact = 2.0 / (p + 1) if p % 2 == 0 else 0.0
        if abs(k - exact) > 1e-10:
            ok_kron = False
    check("Kronrod rule exact for degree <= 22", ok_kron)

    # --- Kronrod outperforms Gauss beyond degree 13 (its extra points earn their keep) ---
    k, g, e = gk.gauss_kronrod_15(lambda x: x ** 16, -1.0, 1.0)
    exact16 = 2.0 / 17
    check("Kronrod beats Gauss at degree 16", abs(k - exact16) < abs(g - exact16))

    # --- error estimate is nonzero beyond Kronrod exactness (degree 24) ---
    k, g, e = gk.gauss_kronrod_15(lambda x: x ** 24, -1.0, 1.0)
    check("nonzero error estimate beyond exactness", e > 0)

    # --- smooth transcendental integrals to machine precision (adaptive) ---
    v, err, ni = gk.integrate(math.sin, 0.0, math.pi, tol=1e-12)
    check("integral of sin on [0,pi] == 2", abs(v - 2.0) < 1e-10)
    check("  error estimate bounds true error (sin)", err >= abs(v - 2.0) or abs(v - 2.0) < 1e-12)

    v, err, ni = gk.integrate(math.exp, 0.0, 1.0, tol=1e-12)
    check("integral of exp on [0,1] == e-1", abs(v - (math.e - 1.0)) < 1e-11)

    v, err, ni = gk.integrate(lambda x: 1.0 / (1.0 + x * x), -1.0, 1.0, tol=1e-12)
    check("integral of 1/(1+x^2) on [-1,1] == pi/2", abs(v - math.pi / 2) < 1e-11)

    # --- adaptive driver resolves a TALL NARROW spike a fixed grid misses ---
    # gaussian bump of width 0.01 centered at 0.3, on [0,1]; true integral ~ sqrt(pi)*0.01
    def spike(x):
        return math.exp(-((x - 0.3) / 0.01) ** 2)
    true_spike = 0.01 * math.sqrt(math.pi)  # tails beyond [0,1] negligible
    v, err, ni = gk.integrate(spike, 0.0, 1.0, tol=1e-10)
    check("adaptive resolves narrow spike", abs(v - true_spike) < 1e-8)
    # a single fixed panel badly under-resolves it
    vfix, _ = gk.integrate_fixed(spike, 0.0, 1.0, panels=1)
    check("single fixed panel misses the spike", abs(vfix - true_spike) > 1e-3)
    check("  adaptive subdivided for the spike", ni > 3)

    # --- endpoint square-root singularity: integral of 1/sqrt(x) on [0,1] = 2 ---
    def invsqrt(x):
        return 1.0 / math.sqrt(x) if x > 0 else 0.0
    v, err, ni = gk.integrate(invsqrt, 0.0, 1.0, tol=1e-8, max_intervals=4000)
    check("adaptive handles endpoint singularity", abs(v - 2.0) < 1e-4)

    # --- cross-check against the repo's Romberg and adaptive Simpson ---
    f = lambda x: math.cos(x) * math.exp(-x)
    v_gk, _, _ = gk.integrate(f, 0.0, 2.0, tol=1e-12)
    v_rom = quadrature.romberg(f, 0.0, 2.0)
    v_sim = quadrature.adaptive_simpson(f, 0.0, 2.0, tol=1e-12)
    check("GK matches Romberg", abs(v_gk - v_rom) < 1e-9)
    check("GK matches adaptive Simpson", abs(v_gk - v_sim) < 1e-9)

    # --- reversing the limits flips the sign ---
    v_fwd, _, _ = gk.integrate(math.sin, 0.0, 1.0, tol=1e-11)
    v_rev, _, _ = gk.integrate(math.sin, 1.0, 0.0, tol=1e-11)
    check("reversing limits flips sign", abs(v_fwd + v_rev) < 1e-10)

    # --- degenerate interval integrates to zero ---
    v_deg, _, _ = gk.integrate(math.exp, 2.0, 2.0)
    check("zero-width interval == 0", v_deg == 0.0)

    # --- the error estimate genuinely bounds the true error on a smooth case ---
    def poly(x):
        return x ** 5 - 3 * x ** 3 + 2 * x  # antiderivative x^6/6 - 3x^4/4 + x^2
    F = lambda x: x ** 6 / 6 - 3 * x ** 4 / 4 + x ** 2
    exact = F(1.5) - F(0.2)
    v, err, ni = gk.integrate(poly, 0.2, 1.5, tol=1e-12)
    check("error estimate >= true error (poly)", err + 1e-14 >= abs(v - exact))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
