"""Tests for horner.py -- polynomial evaluation and root-finding.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Horner is cross-checked against
direct power-sum evaluation; roots are verified by substitution.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import horner as H  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- evaluation -------------------------------------------------------------
c = [2, -6, 2, -1]      # 2x^3 - 6x^2 + 2x - 1
check("horner evaluates correctly", approx(H.horner(c, 3), 5.0))
check("horner matches direct evaluation", approx(H.horner(c, 3), H.direct_eval(c, 3)))
check("horner at 0 is the constant term", approx(H.horner(c, 0), -1.0))
check("horner at 1 is the coefficient sum", approx(H.horner(c, 1), sum(c)))
check("constant polynomial", approx(H.horner([7], 99), 7.0))
check("linear polynomial", approx(H.horner([3, 5], 2), 11.0))     # 3*2+5

# horner == direct over many random polynomials and points
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 8


gen = lcg(1)


def rf(a, b):
    return a + (b - a) * (next(gen) / (1 << 24))


ok = True
for _ in range(2000):
    deg = 1 + next(gen) % 8
    cc = [rf(-5, 5) for _ in range(deg + 1)]
    x = rf(-3, 3)
    if not approx(H.horner(cc, x), H.direct_eval(cc, x), 1e-7 * (1 + abs(H.direct_eval(cc, x)))):
        ok = False
        break
check("horner matches direct over 2000 random polynomials", ok)

# --- synthetic division -----------------------------------------------------
q, r = H.synthetic_division(c, 3)
check("synthetic division remainder equals p(r)", approx(r, H.horner(c, 3)))
check("synthetic division quotient has one lower degree", len(q) == len(c) - 1)
# reconstruct: (x - r) * quotient + remainder == original
def poly_mul_linear_plus(qc, root, rem):
    # (x - root) * qc + rem, as coefficients highest-first
    out = qc + [0.0]
    for i in range(len(qc)):
        out[i + 1] -= root * qc[i]
    out[-1] += rem
    return out
recon = poly_mul_linear_plus(q, 3, r)
check("(x - r) q(x) + rem reconstructs p", all(approx(recon[i], c[i]) for i in range(len(c))))
# dividing by a root gives zero remainder
qr, rr = H.synthetic_division([1, -5, 6], 2)     # root at x=2
check("dividing by a root leaves zero remainder", approx(rr, 0.0))
check("the quotient of x^2-5x+6 by (x-2) is (x-3)", all(approx(a, b) for a, b in zip(qr, [1, -3])))

# --- derivative -------------------------------------------------------------
p, dp = H.evaluate_with_derivative(c, 3)
check("evaluate_with_derivative gives p(x)", approx(p, 5.0))
check("evaluate_with_derivative gives p'(x)", approx(dp, 20.0))   # 6x^2-12x+2 at 3 = 20
check("derivative coefficients are correct", H.derivative_coeffs(c) == [6, -12, 2])
check("derivative of a constant is 0", H.derivative_coeffs([5]) == [0.0])
# the derivative from coeffs matches the Horner-embedded derivative
check("derivative coeffs evaluate to the embedded derivative",
      approx(H.horner(H.derivative_coeffs(c), 3), dp))

# --- Newton root-finding ----------------------------------------------------
r2, it = H.newton_root([1, 0, -2], 1.0)          # x^2 - 2 -> sqrt(2)
check("Newton finds sqrt(2)", approx(r2, 2 ** 0.5, 1e-10))
check("Newton converges quickly", it < 10)
r_cube, _ = H.newton_root([1, 0, 0, -8], 1.0)    # x^3 - 8 -> 2
check("Newton finds the cube root of 8", approx(r_cube, 2.0, 1e-10))
check("a found root evaluates to ~0", approx(H.horner([1, 0, -2], r2), 0.0, 1e-9))
try:
    H.newton_root([1, 0], 0.0)                   # p'(0)=1 for x, actually fine; use flat case
    # a genuinely zero-derivative start: p(x)=x^2 at x=0 has p'=0
    H.newton_root([1, 0, 0], 0.0)
    check("Newton raises on a zero derivative", False)
except ValueError:
    check("Newton raises on a zero derivative", True)

# --- real_roots via deflation ----------------------------------------------
roots = H.real_roots([1, -6, 11, -6])            # (x-1)(x-2)(x-3)
check("finds all three roots of a cubic", [round(x, 6) for x in roots] == [1.0, 2.0, 3.0])
check("finds both roots of a quadratic", [round(x, 6) for x in H.real_roots([1, -5, 6])] == [2.0, 3.0])
check("a single linear root", [round(x, 6) for x in H.real_roots([2, -8])] == [4.0])
# each returned root really is a root
quartic = [1, -10, 35, -50, 24]                  # (x-1)(x-2)(x-3)(x-4)
qroots = H.real_roots(quartic)
check("all returned roots satisfy the polynomial",
      all(approx(H.horner(quartic, r), 0.0, 1e-5) for r in qroots))
check("finds the expected four roots", [round(x) for x in qroots] == [1, 2, 3, 4])

# --- deflate ----------------------------------------------------------------
check("deflate removes a root", H.deflate([1, -5, 6], 2) == [1, -3])


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall horner tests passed")
