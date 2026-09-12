"""Tests for chebyshev: accuracy on smooth functions, geometric convergence, Runge, nodes."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chebyshev import (ChebyshevInterpolant, cheb_nodes, cheb_polynomial,
                       equispaced_interpolate)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- smooth functions approximated to tiny error ---------------------------
ci = ChebyshevInterpolant(math.exp, 16, -1, 1)
check("exp approximated to near machine precision", ci.max_error(math.exp) < 1e-12)

cs = ChebyshevInterpolant(math.sin, 20, 0, 2 * math.pi)
check("sin over [0,2pi] approximated to near machine precision", cs.max_error(math.sin) < 1e-10)

rational = lambda x: 1.0 / (2 + x)
cr = ChebyshevInterpolant(rational, 20, -1, 1)
check("a rational function approximated to tiny error", cr.max_error(rational) < 1e-10)

# --- error shrinks geometrically with degree ------------------------------
def cheb_error(deg):
    return ChebyshevInterpolant(math.cos, deg, -2, 2).max_error(math.cos)

errs = [cheb_error(d) for d in [4, 8, 12, 16]]
check(f"Chebyshev error shrinks with degree ({errs[0]:.2e} -> {errs[-1]:.2e})",
      errs[-1] < errs[0])
check("high-degree Chebyshev error is tiny for cos", errs[-1] < 1e-8)

# --- the Runge phenomenon: Chebyshev stays bounded, equispaced blows up ----
runge = lambda x: 1.0 / (1 + 25 * x * x)
for deg in [10, 16, 24]:
    cheb = ChebyshevInterpolant(runge, deg, -1, 1)
    equi = equispaced_interpolate(runge, deg, -1, 1)
    cheb_err = cheb.max_error(runge)
    equi_err = max(abs(equi(-1 + 2 * i / 500) - runge(-1 + 2 * i / 500)) for i in range(501))
    check(f"deg {deg}: Chebyshev error ({cheb_err:.3f}) << equispaced error ({equi_err:.1f})",
          cheb_err < equi_err)
# and the equispaced error actually grows with degree (the blowup)
e10 = max(abs(equispaced_interpolate(runge, 10, -1, 1)(-1 + 2 * i / 500) - runge(-1 + 2 * i / 500))
          for i in range(501))
e24 = max(abs(equispaced_interpolate(runge, 24, -1, 1)(-1 + 2 * i / 500) - runge(-1 + 2 * i / 500))
          for i in range(501))
check(f"equispaced Runge error grows with degree ({e10:.1f} -> {e24:.1f})", e24 > e10)
# Chebyshev error shrinks with degree on the same function
c10 = ChebyshevInterpolant(runge, 10, -1, 1).max_error(runge)
c24 = ChebyshevInterpolant(runge, 24, -1, 1).max_error(runge)
check(f"Chebyshev Runge error shrinks with degree ({c10:.3f} -> {c24:.3f})", c24 < c10)

# --- interpolating a low-degree polynomial recovers it exactly -------------
poly = lambda x: 3 * x ** 3 - 2 * x ** 2 + x - 5
cp = ChebyshevInterpolant(poly, 5, -1, 1)
check("degree-5 fit recovers a cubic exactly", cp.max_error(poly) < 1e-10)
# the higher coefficients should be ~0
check("coefficients above the polynomial degree vanish",
      all(abs(cp.coeffs[j]) < 1e-9 for j in range(4, len(cp.coeffs))))

# --- Chebyshev nodes lie in the interval, clustered at the ends ------------
nodes = cheb_nodes(11, -1, 1)
check("nodes lie within the interval", all(-1 <= x <= 1 for x in nodes))
check("nodes are sorted", nodes == sorted(nodes))
check("nodes include the endpoints", abs(nodes[0] + 1) < 1e-9 and abs(nodes[-1] - 1) < 1e-9)
# clustering: the gap near the ends is smaller than in the middle
end_gap = nodes[1] - nodes[0]
mid = len(nodes) // 2
mid_gap = nodes[mid + 1] - nodes[mid]
check("nodes cluster near the ends (end gap < middle gap)", end_gap < mid_gap)

# --- Chebyshev polynomial values -------------------------------------------
check("T_0 = 1", abs(cheb_polynomial(0, 0.5) - 1) < 1e-12)
check("T_1(x) = x", abs(cheb_polynomial(1, 0.5) - 0.5) < 1e-12)
check("T_2(x) = 2x^2 - 1", abs(cheb_polynomial(2, 0.5) - (2 * 0.25 - 1)) < 1e-12)
check("T_n is bounded by 1 on [-1,1]",
      all(abs(cheb_polynomial(7, -1 + 2 * i / 100)) <= 1 + 1e-9 for i in range(101)))

# --- interpolant matches the function AT the nodes exactly -----------------
f = lambda x: math.exp(x) * math.sin(3 * x)
ci = ChebyshevInterpolant(f, 25, -1, 1)
check("high-degree fit of exp*sin is accurate", ci.max_error(f) < 1e-8)

# --- a shifted interval ----------------------------------------------------
g = lambda x: math.log(x)
cg = ChebyshevInterpolant(g, 25, 1, 10)
check("log on [1,10] approximated well", cg.max_error(g) < 1e-7)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all chebyshev tests passed")
