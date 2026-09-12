"""Tests for durand_kerner: all-roots finding vs known roots, residuals, and Vieta reconstruction."""

import os
import sys
import cmath

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from durand_kerner import roots, real_roots, from_roots, match_roots, residual

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randf(self, lo, hi):
        return lo + (self.rand() / 65536.0) * (hi - lo)


# --- known factorisations --------------------------------------------------
check("x^2-3x+2 has roots {1,2}", sorted(round(z.real, 6) for z in roots([1, -3, 2])) == [1.0, 2.0])
check("x^3-6x^2+11x-6 has real roots {1,2,3}",
      [round(r, 6) for r in real_roots([1, -6, 11, -6])] == [1.0, 2.0, 3.0])
check("x^2+1 has roots +-i",
      match_roots(roots([1, 0, 1]), [1j, -1j]))
check("linear x-5 has root 5", match_roots(roots([1, -5]), [5]))
check("x^4-1 has roots 1,-1,i,-i", match_roots(roots([1, 0, 0, 0, -1]), [1, -1, 1j, -1j]))

# repeated roots: (x-2)^3
check("(x-2)^3 has a triple root at 2",
      match_roots(roots(from_roots([2, 2, 2])), [2, 2, 2], tol=1e-3))

# degree 0 / constant
check("constant polynomial has no roots", roots([7]) == [])

# leading zeros are stripped
check("leading zeros stripped: 0x^2 + x - 4 has root 4",
      match_roots(roots([0, 1, -4]), [4]))

# --- build from known roots, recover them ----------------------------------
rng = LCG(2026)
recover_ok = True
for _ in range(300):
    n = 1 + rng.rand() % 5
    known = []
    # mix of real and complex-conjugate pairs
    while len(known) < n:
        if rng.rand() % 2 == 0 or len(known) == n - 1:
            known.append(complex(round(rng.randf(-5, 5), 3), 0))
        else:
            a = round(rng.randf(-4, 4), 3)
            b = round(rng.randf(0.3, 4), 3)
            known.append(complex(a, b))
            known.append(complex(a, -b))
    known = known[:n]
    coeffs = from_roots(known)
    found = roots(coeffs)
    if not match_roots(found, known, tol=1e-4):
        recover_ok = False
        print(f"  failed to recover roots {known}")
        break
check("roots recovered from a polynomial built out of known roots (300 polys)", recover_ok)

# --- residuals are ~0 for every returned root ------------------------------
rng = LCG(4242)
resid_ok = True
for _ in range(300):
    deg = 1 + rng.rand() % 5
    coeffs = [complex(round(rng.randf(-6, 6), 3), round(rng.randf(-6, 6), 3)) for _ in range(deg + 1)]
    if coeffs[0] == 0:
        coeffs[0] = 1
    rs = roots(coeffs)
    if len(rs) != deg:
        resid_ok = False
        break
    for r in rs:
        if residual(coeffs, r) > 1e-5:
            resid_ok = False
            print(f"  large residual {residual(coeffs, r)} for coeffs={coeffs}")
            break
    if not resid_ok:
        break
check("every returned root has ~zero polynomial residual (300 random polys)", resid_ok)

# --- Vieta: elementary symmetric functions of the roots reproduce coeffs ---
rng = LCG(777)
vieta_ok = True
for _ in range(200):
    n = 1 + rng.rand() % 4
    known = [complex(round(rng.randf(-4, 4), 3), round(rng.randf(-4, 4), 3)) for _ in range(n)]
    coeffs = from_roots(known)          # monic
    rs = roots(coeffs)
    # sum of roots == -coeffs[1] (monic), product == (-1)^n coeffs[n]
    sum_roots = sum(rs)
    prod_roots = 1 + 0j
    for r in rs:
        prod_roots *= r
    if abs(sum_roots - (-coeffs[1])) > 1e-4:
        vieta_ok = False
        break
    if abs(prod_roots - ((-1) ** n) * coeffs[n]) > 1e-4:
        vieta_ok = False
        break
check("Vieta's formulas: root sum and product match the coefficients (200 polys)", vieta_ok)

# --- correct number of roots (with multiplicity) ---------------------------
rng = LCG(555)
count_ok = True
for _ in range(200):
    deg = 1 + rng.rand() % 6
    coeffs = [1] + [round(rng.randf(-5, 5), 3) for _ in range(deg)]
    if len(roots(coeffs)) != deg:
        count_ok = False
        break
check("a degree-n polynomial returns exactly n roots", count_ok)

# --- real_roots filters correctly ------------------------------------------
# x^3 - x = x(x-1)(x+1): three real roots; x^2+1: none
check("real_roots of x^3-x are {-1,0,1}",
      [round(r, 6) for r in real_roots([1, 0, -1, 0])] == [-1.0, 0.0, 1.0])
check("real_roots of x^2+1 is empty", real_roots([1, 0, 1]) == [])

# --- Wilkinson-ish: (x-1)(x-2)...(x-5) recovers 1..5 -----------------------
poly = from_roots([1, 2, 3, 4, 5])
check("product (x-1)...(x-5) recovers roots 1..5",
      [round(r) for r in real_roots(poly)] == [1, 2, 3, 4, 5])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all durand_kerner tests passed")
