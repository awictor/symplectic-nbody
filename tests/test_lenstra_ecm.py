"""Tests for Lenstra ECM factorization: divisibility, product = N, primality of factors, hard cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lenstra_ecm as E  # noqa: E402
from baillie_psw import is_prime  # noqa: E402


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
    # ---- 1. semiprimes factor correctly -------------------------------------------------
    for N in (1073, 8051, 10403, 1000003 * 1000033):
        f = E.factorize(N)
        check(f"factor {N}: product == N and factors prime", E.verify(N, f), f"{f}")

    # ---- 2. every returned factor divides N ---------------------------------------------
    N = 600851475143       # Project Euler #3 famous number
    f = E.factorize(N)
    check("all factors divide N", all(N % p == 0 for p in f) and E.verify(N, f), f"{f}")
    check("largest prime factor of 600851475143 is 6857", max(f) == 6857, f"{f}")

    # ---- 3. prime powers ----------------------------------------------------------------
    check("2^10 -> ten 2s", E.factorize(2 ** 10) == [2] * 10)
    check("3^7 -> seven 3s", E.factorize(3 ** 7) == [3] * 7)
    check("prime power 7^5", E.factorize(7 ** 5) == [7] * 5)

    # ---- 4. a prime is its own sole factor ----------------------------------------------
    check("1000003 is prime -> [1000003]", E.factorize(1000003) == [1000003])
    check("9999999967 is prime -> singleton", E.factorize(9999999967) == [9999999967])

    # ---- 5. numbers with several distinct prime factors ---------------------------------
    N = 2 * 3 * 5 * 7 * 11 * 13 * 17
    check("primorial factors", E.factorize(N) == [2, 3, 5, 7, 11, 13, 17])
    N2 = 2 ** 3 * 3 ** 2 * 5 * 101
    check("mixed multiplicities", sorted(E.factorize(N2)) == sorted([2, 2, 2, 3, 3, 5, 101]))

    # ---- 6. a 10-digit-prime semiprime (bigger than trial division likes) ---------------
    N = 1000000007 * 1000000009
    f = E.factorize(N)
    check("large semiprime factored", sorted(f) == [1000000007, 1000000009] and E.verify(N, f), f"{f}")

    # ---- 7. find_factor returns a genuine non-trivial divisor ---------------------------
    N = 8051
    fac = E.find_factor(N)
    check("find_factor returns a proper divisor", fac is not None and 1 < fac < N and N % fac == 0,
          f"{fac}")

    # ---- 8. a case where Pollard p-1 is weak: p-1 and q-1 both have a large prime factor -
    # p = 2*15485863 + 1 style; use two primes p,q with p-1, q-1 non-smooth
    p = 1000000007   # p-1 = 2 * 500000003 (500000003 prime -> not smooth)
    q = 1000000009
    N = p * q
    f = E.factorize(N)
    check("factors a semiprime with non-smooth p-1 (ECM strength)", E.verify(N, f), f"{f}")

    # ---- 9. every factor passes the Baillie-PSW primality test --------------------------
    N = 13 * 17 * 19 * 9999999967
    f = E.factorize(N)
    check("all returned factors are prime (Baillie-PSW)", all(is_prime(p) for p in f) and E.verify(N, f),
          f"{f}")

    # ---- 10. edge cases -----------------------------------------------------------------
    check("factorize(1) == []", E.factorize(1) == [])
    check("factorize(2) == [2]", E.factorize(2) == [2])
    check("even number peels 2s", E.factorize(2 * 2 * 3 * 3 * 3) == [2, 2, 3, 3, 3])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
