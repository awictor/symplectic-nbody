"""Tests for continued_fraction: convergents, roundtrip, best-approximation property, known CFs."""

import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from continued_fraction import (cf_expansion, convergents, evaluate, best_approximation,
                                approximation_error)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- pi's famous convergents -----------------------------------------------
c = convergents(cf_expansion(math.pi, 12))
check("pi's first convergent is 3", c[0] == Fraction(3))
check("pi's convergent 22/7", c[1] == Fraction(22, 7))
check("pi's convergent 333/106", c[2] == Fraction(333, 106))
check("pi's convergent 355/113", c[3] == Fraction(355, 113))

# --- an exact fraction round-trips through its CF --------------------------
ok = True
for num, den in [(355, 113), (22, 7), (1, 3), (100, 7), (7, 100), (12345, 6789)]:
    f = Fraction(num, den)
    if evaluate(cf_expansion(f)) != f:
        ok = False
        break
check("exact fraction recovered from its continued fraction", ok)

# --- the golden ratio is all ones -----------------------------------------
phi = (1 + math.sqrt(5)) / 2
check("golden ratio expansion is all 1s", cf_expansion(phi, 15) == [1] * 15)

# --- sqrt(2) is [1; 2, 2, 2, ...] ------------------------------------------
s2 = cf_expansion(math.sqrt(2), 12)
check("sqrt(2) is [1; 2, 2, ...]", s2[0] == 1 and all(a == 2 for a in s2[1:]))

# --- sqrt(3) is [1; 1, 2, 1, 2, ...] (periodic) ----------------------------
s3 = cf_expansion(math.sqrt(3), 12)
check("sqrt(3) is [1; 1, 2, 1, 2, ...]", s3[0] == 1 and s3[1:5] == [1, 2, 1, 2])

# --- convergents are in lowest terms ---------------------------------------
c = convergents(cf_expansion(math.pi, 10))
check("convergents are in lowest terms",
      all(math.gcd(f.numerator, f.denominator) == 1 for f in c))

# --- convergents alternate around the target and converge ------------------
target = math.pi
c = convergents(cf_expansion(target, 10))
errs = [approximation_error(target, f) for f in c]
check("convergent errors are strictly decreasing", all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)))
# alternate above/below
signs = [1 if float(f) > target else -1 for f in c]
check("convergents alternate above and below the target",
      all(signs[i] != signs[i + 1] for i in range(len(signs) - 1)))

# --- the best-approximation property ---------------------------------------
# each convergent p/q is the closest fraction to x among all fractions with denominator <= q
def is_best_for_denominator(x, frac):
    q = frac.denominator
    my_err = abs(x - float(frac))
    for d in range(1, q + 1):
        # nearest numerator for this denominator
        best_n = round(x * d)
        err = abs(x - best_n / d)
        if err < my_err - 1e-12:
            return False
    return True


# (the 0th convergent is the floor a0, not the nearest integer, so the best-approximation
#  theorem applies from the 1st convergent on -- skip index 0.)
c = convergents(cf_expansion(math.pi, 6))
check("each pi convergent (from the 1st) is the best approximation for its denominator",
      all(is_best_for_denominator(math.pi, f) for f in c[1:] if f.denominator <= 200))

# also for e
c_e = convergents(cf_expansion(math.e, 8))
check("each e convergent (from the 1st) is the best for its denominator",
      all(is_best_for_denominator(math.e, f) for f in c_e[1:] if f.denominator <= 200))

# --- best_approximation within a denominator bound -------------------------
check("best rational for pi with den<=113 is 355/113",
      best_approximation(math.pi, 113) == Fraction(355, 113))
check("best rational for pi with den<=10 is 22/7",
      best_approximation(math.pi, 10) == Fraction(22, 7))
check("best rational for pi with den<=1 is 3",
      best_approximation(math.pi, 1) == Fraction(3))

# best_approximation is genuinely best within the bound
def brute_best(x, maxd):
    best = None
    best_err = float("inf")
    for d in range(1, maxd + 1):
        n = round(x * d)
        e = abs(x - n / d)
        if e < best_err:
            best_err = e
            best = Fraction(n, d)
    return best


for maxd in [7, 50, 113, 200]:
    ba = best_approximation(math.pi, maxd)
    bb = brute_best(math.pi, maxd)
    check(f"best_approximation matches brute force (den<={maxd})",
          abs(math.pi - float(ba)) <= abs(math.pi - float(bb)) + 1e-12)

# --- e's continued fraction has the known pattern [2;1,2,1,1,4,1,1,6,...] --
e_cf = cf_expansion(math.e, 11)
check("e's CF starts [2; 1, 2, 1, 1, 4, 1, 1, 6]", e_cf[:9] == [2, 1, 2, 1, 1, 4, 1, 1, 6])

# --- a rational's CF is finite and evaluates back exactly ------------------
check("rational 649/200 round-trips", evaluate(cf_expansion(Fraction(649, 200))) == Fraction(649, 200))

# --- integer input ---------------------------------------------------------
check("integer 5 has CF [5]", cf_expansion(Fraction(5)) == [5])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all continued_fraction tests passed")
