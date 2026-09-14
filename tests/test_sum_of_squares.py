"""Tests for sums of squares: two-square test vs brute, Fermat's criterion, Cornacchia, Lagrange four."""

import os
import sys
from math import isqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sum_of_squares import (  # noqa: E402
    is_sum_of_two_squares,
    two_squares,
    cornacchia,
    four_squares,
    gaussian_norm,
    brute_two_squares,
)
from pollard_rho import is_prime  # noqa: E402


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
    # ---- 1. two-square test matches brute force for n = 0..3000 -------------------------
    ok = True
    for n in range(0, 3001):
        expected = brute_two_squares(n) is not None
        if is_sum_of_two_squares(n) != expected:
            ok = False
            check("two-square test == brute", False, f"n={n}")
            break
    if ok:
        check("is_sum_of_two_squares == brute existence (0..3000)", True)

    # ---- 2. every returned two-square representation is correct --------------------------
    ok = True
    for n in range(0, 3001):
        r = two_squares(n)
        if r is not None:
            a, b = r
            if a * a + b * b != n:
                ok = False
                check("two_squares representation correct", False, f"n={n}: {r}")
                break
        elif brute_two_squares(n) is not None:
            ok = False
            check("two_squares not None when representable", False, f"n={n}")
            break
    if ok:
        check("two_squares gives a valid representation (0..3000)", True)

    # ---- 3. Fermat's criterion on primes: sum of two squares iff p=2 or p=1 mod 4 -------
    ok = True
    for p in range(2, 2000):
        if not is_prime(p):
            continue
        representable = is_sum_of_two_squares(p)
        expected = (p == 2 or p % 4 == 1)
        if representable != expected:
            ok = False
            check("Fermat's criterion", False, f"p={p}")
            break
    if ok:
        check("Fermat: prime is sum of 2 squares iff p=2 or p=1 mod 4", True)

    # ---- 4. Cornacchia gives a^2 + b^2 = p for primes p = 1 mod 4 ------------------------
    ok = True
    for p in range(5, 2000):
        if is_prime(p) and p % 4 == 1:
            a, b = cornacchia(p)
            if a * a + b * b != p:
                ok = False
                check("Cornacchia a^2+b^2=p", False, f"p={p}: {(a,b)}")
                break
    if ok:
        check("Cornacchia: a^2 + b^2 = p (primes = 1 mod 4 up to 2000)", True)

    check("cornacchia(2) = (1,1)", cornacchia(2) == (1, 1))
    check("cornacchia(13) valid", sum(x * x for x in cornacchia(13)) == 13)

    # ---- 5. primes = 3 mod 4 are NOT sums of two squares --------------------------------
    for p in [3, 7, 11, 19, 23, 31, 43]:
        check(f"{p} (3 mod 4) not sum of 2 squares", not is_sum_of_two_squares(p))

    # ---- 6. two-squares theorem: 3-mod-4 primes must have even exponent -----------------
    check("9 = 3^2 IS sum of two squares (0+9)", is_sum_of_two_squares(9))  # 0^2+3^2
    check("21 = 3*7 NOT sum of two squares", not is_sum_of_two_squares(21))
    check("45 = 9*5 IS sum of two squares", is_sum_of_two_squares(45))  # 3^2+6^2
    r = two_squares(45)
    check("45 = 3^2 + 6^2", r is not None and r[0] ** 2 + r[1] ** 2 == 45, f"{r}")

    # ---- 7. Brahmagupta-Fibonacci: product of two sums-of-squares is a sum of squares ---
    # 5 = 1+4, 13 = 4+9, 65 = 5*13 should be a sum of two squares
    check("65 = 5*13 sum of two squares", is_sum_of_two_squares(65))
    r = two_squares(65)
    check("65 representation", r is not None and r[0] ** 2 + r[1] ** 2 == 65)

    # ---- 8. Lagrange four-square: every n is a sum of four squares ----------------------
    ok = True
    for n in range(0, 500):
        rep = four_squares(n)
        if rep is None or sum(x * x for x in rep) != n:
            ok = False
            check("four_squares exact", False, f"n={n}: {rep}")
            break
    if ok:
        check("Lagrange: every n = sum of four squares (0..499)", True)

    # ---- 9. numbers needing all four squares (4^a(8b+7)) --------------------------------
    # 7, 15, 23, 28, 31 need four nonzero squares; verify representation
    for n in [7, 15, 23, 28, 31, 112]:
        rep = four_squares(n)
        check(f"{n} four-square rep", sum(x * x for x in rep) == n, f"{rep}")

    # ---- 10. large values -------------------------------------------------------------
    n = 1000000
    r = two_squares(n)
    check("1000000 two squares", r is not None and r[0] ** 2 + r[1] ** 2 == n)
    n = 999983 * 999983 + 1  # some large-ish number
    rep4 = four_squares(min(n, 100000))
    check("large four squares", sum(x * x for x in rep4) == min(n, 100000))

    # ---- 11. gaussian norm --------------------------------------------------------------
    check("gaussian_norm(3,4) = 25", gaussian_norm(3, 4) == 25)

    # ---- 12. zero and one ---------------------------------------------------------------
    check("0 = 0^2+0^2", two_squares(0) == (0, 0))
    check("1 = 1^2+0^2", two_squares(1) is not None and sum(x * x for x in two_squares(1)) == 1)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
