"""Tests for Stirling/Bell numbers: brute enumeration, recurrences, row-sum identities, duality."""

import os
import sys
from math import factorial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stirling import (  # noqa: E402
    stirling_second,
    stirling_first,
    signed_stirling_first,
    bell,
    bell_triangle,
    stirling_second_row,
    stirling_first_row,
    falling_factorial_coeffs,
    power_in_falling_factorials,
    bell_dobinski,
    brute_stirling_second,
    brute_stirling_first,
    brute_bell,
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
    # ---- 1. S(n,k) matches brute set-partition count ------------------------------------
    mism = 0
    for n in range(0, 9):
        for k in range(0, n + 1):
            if stirling_second(n, k) != brute_stirling_second(n, k):
                mism += 1
    check("S(n,k) == brute set-partition count (n<=8)", mism == 0, f"{mism}")

    # ---- 2. c(n,k) matches brute permutation-cycle count --------------------------------
    mism = 0
    for n in range(0, 8):
        for k in range(0, n + 1):
            if stirling_first(n, k) != brute_stirling_first(n, k):
                mism += 1
    check("c(n,k) == brute cycle count (n<=7)", mism == 0, f"{mism}")

    # ---- 3. Bell numbers ----------------------------------------------------------------
    known_bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    check("Bell numbers match known values", [bell(n) for n in range(11)] == known_bell)
    check("B(n) == brute total partitions (n<=8)", all(bell(n) == brute_bell(n) for n in range(9)))

    # ---- 4. row-sum identities ----------------------------------------------------------
    check("sum_k S(n,k) == B(n)", all(sum(stirling_second_row(n)) == bell(n) for n in range(0, 11)))
    check("sum_k c(n,k) == n!", all(sum(stirling_first_row(n)) == factorial(n) for n in range(0, 9)))

    # ---- 5. known small values ----------------------------------------------------------
    check("S(4,2) = 7", stirling_second(4, 2) == 7)
    check("S(n,1) = 1", all(stirling_second(n, 1) == 1 for n in range(1, 8)))
    check("S(n,n) = 1", all(stirling_second(n, n) == 1 for n in range(0, 8)))
    check("S(n,2) = 2^(n-1) - 1", all(stirling_second(n, 2) == 2 ** (n - 1) - 1 for n in range(2, 9)))
    check("c(n,1) = (n-1)!", all(stirling_first(n, 1) == factorial(n - 1) for n in range(1, 8)))
    check("c(n,n) = 1", all(stirling_first(n, n) == 1 for n in range(0, 8)))

    # ---- 6. Bell triangle left edge = Bell numbers --------------------------------------
    tri = bell_triangle(10)
    check("Bell triangle left edge = Bell numbers", [row[0] for row in tri] == [bell(n) for n in range(10)])
    check("Bell triangle right edge = next Bell", all(tri[i][-1] == bell(i + 1) for i in range(9)))

    # ---- 7. Dobinski's formula ----------------------------------------------------------
    check("Dobinski series matches Bell numbers",
          all(abs(bell_dobinski(n) - bell(n)) < 1e-6 for n in range(0, 8)))

    # ---- 8. Stirling duality: the two kinds are inverse matrices ------------------------
    # sum_k s(n,k) S(k,m) = [n==m] (Kronecker delta), using signed first kind
    def kron(n, m):
        total = 0
        for k in range(min(n, m), n + 1):  # only k where both nonzero
            total += signed_stirling_first(n, k) * stirling_second(k, m)
        return total
    ok = all(kron(n, m) == (1 if n == m else 0) for n in range(0, 7) for m in range(0, n + 1))
    check("signed first & second kind are inverse matrices", ok)

    # ---- 9. falling-factorial polynomial identities -------------------------------------
    # (x)_n evaluated via coeffs == product x(x-1)...(x-n+1) at integer x
    def poly_eval(coeffs, x):
        return sum(c * x ** i for i, c in enumerate(coeffs))

    def falling(x, n):
        p = 1
        for i in range(n):
            p *= (x - i)
        return p
    ok = True
    for n in range(0, 7):
        coeffs = falling_factorial_coeffs(n)
        for x in range(-3, 8):
            if poly_eval(coeffs, x) != falling(x, n):
                ok = False
    check("falling-factorial coeffs = signed first kind", ok)

    # x^n = sum_k S(n,k) (x)_k
    ok = True
    for n in range(0, 7):
        S = power_in_falling_factorials(n)
        for x in range(-2, 6):
            val = sum(S[k] * falling(x, k) for k in range(n + 1))
            if val != x ** n:
                ok = False
    check("x^n = sum_k S(n,k) (x)_k", ok)

    # ---- 10. edge cases -----------------------------------------------------------------
    check("S(0,0) = 1", stirling_second(0, 0) == 1)
    check("B(0) = 1", bell(0) == 1)
    check("S(n,0) = 0 for n>0", all(stirling_second(n, 0) == 0 for n in range(1, 6)))
    check("S(n,k) = 0 for k>n", stirling_second(3, 5) == 0)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
