"""Tests for rootfind.py -- bracketing root-finders.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Each method is checked against
known roots and the four are checked to agree.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rootfind as R  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


sqrt2 = 2 ** 0.5
f = lambda x: x * x - 2

# --- bisection --------------------------------------------------------------
r, it = R.bisection(f, 0, 2)
check("bisection finds sqrt(2)", approx(r, sqrt2, 1e-11))
check("bisection returns an iteration count", it > 0)
check("bisection detects an exact root at an endpoint", R.bisection(lambda x: x - 3, 3, 5)[0] == 3)
check("bisection on a linear function", approx(R.bisection(lambda x: 2 * x - 6, 0, 10)[0], 3.0))
try:
    R.bisection(f, 3, 4)      # no sign change
    check("bisection needs a bracket", False)
except ValueError:
    check("bisection needs a bracket", True)

# --- secant -----------------------------------------------------------------
rs, its = R.secant(f, 0, 2)
check("secant finds sqrt(2)", approx(rs, sqrt2, 1e-10))
check("secant is fast (few iterations)", its < 15)
check("secant on a cubic root", approx(R.secant(lambda x: x ** 3 - 8, 1, 3)[0], 2.0, 1e-9))

# --- false position ---------------------------------------------------------
rfp, itfp = R.false_position(f, 0, 2)
check("false position finds sqrt(2)", approx(rfp, sqrt2, 1e-9))
check("false position stays bracketed and converges", itfp < R.bisection(f, 0, 2)[1])
try:
    R.false_position(f, 3, 4)
    check("false position needs a bracket", False)
except ValueError:
    check("false position needs a bracket", True)

# --- Brent ------------------------------------------------------------------
rb, itb = R.brent(f, 0, 2)
check("Brent finds sqrt(2)", approx(rb, sqrt2, 1e-11))
check("Brent detects an exact endpoint root", R.brent(lambda x: x - 3, 3, 5)[0] == 3)
check("Brent on a transcendental (cos x = x)", approx(R.brent(lambda x: math.cos(x) - x, 0, 1)[0], 0.7390851332151607, 1e-9))
try:
    R.brent(f, 3, 4)
    check("Brent needs a bracket", False)
except ValueError:
    check("Brent needs a bracket", True)

# --- all four agree on the same root ---------------------------------------
poly = lambda x: (x - 1.3) * (x - 2.7) * (x + 0.5)
b_r = R.bisection(poly, 2, 3)[0]
s_r = R.secant(poly, 2, 3)[0]
fp_r = R.false_position(poly, 2, 3)[0]
br_r = R.brent(poly, 2, 3)[0]
check("all methods find the same root (2.7)",
      all(approx(x, 2.7, 1e-8) for x in (b_r, s_r, fp_r, br_r)))

# --- roots of transcendental functions -------------------------------------
check("Brent finds a root of e^x - 2 (ln 2)", approx(R.brent(lambda x: math.exp(x) - 2, 0, 1)[0], math.log(2), 1e-10))
check("Brent finds pi as a root of sin", approx(R.brent(math.sin, 3, 4)[0], math.pi, 1e-10))
check("Brent finds a root of x - e^-x", approx(R.brent(lambda x: x - math.exp(-x), 0, 1)[0], 0.5671432904097838, 1e-9))

# --- bracket helpers --------------------------------------------------------
check("has_bracket detects a sign change", R.has_bracket(f, 0, 2))
check("has_bracket is False without one", not R.has_bracket(f, 3, 4))
brs = R.find_brackets(math.sin, 0.5, 10, 200)
check("find_brackets locates sin's roots", len(brs) == 3)     # pi, 2pi, 3pi in (0.5, 10)
roots = [R.brent(math.sin, a, b)[0] for a, b in brs]
check("bracketed sin roots refine to n*pi",
      all(approx(rt, math.pi * (k + 1), 1e-9) for k, rt in enumerate(roots)))

# --- convergence-order sanity: secant/Brent beat bisection -----------------
# on a smooth function, secant should need far fewer iterations than bisection
check("secant needs fewer iterations than bisection", R.secant(f, 0, 2)[1] < R.bisection(f, 0, 2)[1])

# --- a polynomial with several roots via find_brackets + brent -------------
p = lambda x: x ** 3 - 6 * x ** 2 + 11 * x - 6    # roots 1, 2, 3
# use an odd subinterval count so grid points miss the integer roots (avoids a root landing
# exactly on a boundary and being reported by two adjacent brackets)
raw = sorted(R.brent(p, a, b)[0] for a, b in R.find_brackets(p, 0.5, 3.5, 301))
found = []
for r in raw:
    if not found or abs(r - found[-1]) > 1e-6:
        found.append(round(r, 6))
check("finds all three polynomial roots", found == [1.0, 2.0, 3.0])


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall rootfind tests passed")
