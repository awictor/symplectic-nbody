"""Tests for quadrature.py -- numerical integration.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Every method is checked against
integrals with known exact values, and the convergence orders are verified.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import quadrature as Q  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- x^2 on [0, 3] = 9 ------------------------------------------------------
check("trapezoid integrates x^2", approx(Q.trapezoid(lambda x: x * x, 0, 3, 2000), 9.0, 1e-4))
check("simpson integrates x^2 exactly (quadratic)", approx(Q.simpson(lambda x: x * x, 0, 3, 10), 9.0, 1e-10))
check("romberg integrates x^2", approx(Q.romberg(lambda x: x * x, 0, 3), 9.0, 1e-10))
check("adaptive simpson integrates x^2", approx(Q.adaptive_simpson(lambda x: x * x, 0, 3), 9.0, 1e-9))
check("gauss-legendre integrates x^2", approx(Q.gauss_legendre(lambda x: x * x, 0, 3), 9.0, 1e-10))

# --- Simpson and Gauss are exact for low-degree polynomials ----------------
check("simpson is exact for a cubic", approx(Q.simpson(lambda x: x ** 3, 0, 2, 2), 4.0, 1e-12))
check("gauss-5 is exact up to degree 9", approx(Q.gauss_legendre(lambda x: x ** 9, 0, 1, 1), 0.1, 1e-10))
check("gauss-5 is exact for degree 8", approx(Q.gauss_legendre(lambda x: x ** 8, 0, 1, 1), 1 / 9, 1e-10))

# --- transcendental integrals with known values ---------------------------
check("integral of sin on [0, pi] is 2", approx(Q.romberg(math.sin, 0, math.pi), 2.0, 1e-10))
check("integral of exp on [0, 1] is e - 1", approx(Q.gauss_legendre(math.exp, 0, 1, 4), math.e - 1, 1e-10))
check("integral of 1/x on [1, e] is 1", approx(Q.romberg(lambda x: 1 / x, 1, math.e), 1.0, 1e-10))
check("integral of cos on [0, pi/2] is 1", approx(Q.adaptive_simpson(math.cos, 0, math.pi / 2), 1.0, 1e-10))
# the Gaussian bell integral ~ sqrt(2 pi) over a wide interval
check("gaussian integral ~ sqrt(2 pi)", approx(Q.romberg(lambda x: math.exp(-x * x / 2), -6, 6), math.sqrt(2 * math.pi), 1e-6))

# --- convergence orders -----------------------------------------------------
e_trap_50 = abs(Q.trapezoid(math.sin, 0, math.pi, 50) - 2.0)
e_trap_100 = abs(Q.trapezoid(math.sin, 0, math.pi, 100) - 2.0)
check("trapezoid error drops ~4x when n doubles (O(h^2))", approx(e_trap_50 / e_trap_100, 4.0, 0.2))
e_simp_50 = abs(Q.simpson(math.sin, 0, math.pi, 50) - 2.0)
e_simp_100 = abs(Q.simpson(math.sin, 0, math.pi, 100) - 2.0)
check("simpson error drops ~16x when n doubles (O(h^4))", approx(e_simp_50 / e_simp_100, 16.0, 1.0))

# --- accuracy per evaluation: higher-order beats trapezoid -----------------
target = 2.0
check("Simpson beats trapezoid at the same n",
      abs(Q.simpson(math.sin, 0, math.pi, 20) - target) < abs(Q.trapezoid(math.sin, 0, math.pi, 20) - target))
check("Romberg reaches machine precision on a smooth integrand",
      abs(Q.romberg(math.sin, 0, math.pi) - target) < 1e-12)

# --- adaptive concentrates on hard regions ---------------------------------
# a sharply peaked integrand: adaptive Simpson should still nail it
peak = lambda x: math.exp(-100 * (x - 0.5) ** 2)     # narrow spike near 0.5
exact_peak = math.sqrt(math.pi / 100)                # integral over all reals
check("adaptive Simpson handles a sharp peak", approx(Q.adaptive_simpson(peak, 0, 1, tol=1e-12), exact_peak, 1e-8))

# --- edge cases -------------------------------------------------------------
check("zero-width interval integrates to 0", approx(Q.simpson(math.sin, 1.0, 1.0, 4), 0.0, 1e-12))
check("reversed limits negate the integral",
      approx(Q.romberg(math.sin, math.pi, 0), -2.0, 1e-10))
check("a constant integrand gives (b-a)*c", approx(Q.gauss_legendre(lambda x: 5.0, 0, 4, 1), 20.0, 1e-10))
try:
    Q.trapezoid(math.sin, 0, 1, 0)
    check("trapezoid rejects n < 1", False)
except ValueError:
    check("trapezoid rejects n < 1", True)
try:
    Q.gauss_legendre(math.sin, 0, 1, 0)
    check("gauss rejects panels < 1", False)
except ValueError:
    check("gauss rejects panels < 1", True)

# --- all methods agree on a nontrivial integral ----------------------------
g = lambda x: math.exp(x) * math.cos(x)              # antiderivative e^x(sin+cos)/2
exact = (math.exp(1) * (math.sin(1) + math.cos(1)) - 1) / 2
methods = [Q.trapezoid(g, 0, 1, 4000), Q.simpson(g, 0, 1, 200),
           Q.romberg(g, 0, 1), Q.adaptive_simpson(g, 0, 1), Q.gauss_legendre(g, 0, 1, 4)]
check("all five methods agree on integral of e^x cos x",
      all(approx(m, exact, 1e-5) for m in methods))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall quadrature tests passed")
