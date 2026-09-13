"""Tests for Fibonacci: fast doubling == naive/matrix, Cassini, GCD property, Pisano, Zeckendorf."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fibonacci import (  # noqa: E402
    fibonacci, lucas, fib_mod, pisano_period, zeckendorf, cassini, fib_index,
    naive_fibonacci, matrix_fibonacci, fib_pair,
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
    # ---- 1. fast doubling == naive for n = 0..1000 --------------------------------------
    ok = all(fibonacci(n) == naive_fibonacci(n) for n in range(1001))
    check("fast doubling == naive (n=0..1000)", ok)

    # ---- 2. fast doubling == matrix power ------------------------------------------------
    ok = all(fibonacci(n) == matrix_fibonacci(n) for n in range(200))
    check("fast doubling == matrix power (n=0..199)", ok)

    # ---- 3. base cases and a large value -------------------------------------------------
    check("F_0=0, F_1=1, F_10=55", fibonacci(0) == 0 and fibonacci(1) == 1 and fibonacci(10) == 55)
    check("F_100 correct", fibonacci(100) == 354224848179261915075)

    # ---- 4. Lucas numbers ---------------------------------------------------------------
    check("L_0=2, L_1=1, L_2=3, L_10=123",
          lucas(0) == 2 and lucas(1) == 1 and lucas(2) == 3 and lucas(10) == 123)
    # identity L_n = F_{n-1} + F_{n+1}
    check("L_n = F_{n-1}+F_{n+1}", all(lucas(n) == fibonacci(n - 1) + fibonacci(n + 1)
                                       for n in range(1, 50)))

    # ---- 5. Cassini's identity F_{n-1}F_{n+1} - F_n^2 = (-1)^n ---------------------------
    ok = all(cassini(n) == (-1) ** n for n in range(1, 100))
    check("Cassini's identity holds (n=1..99)", ok)

    # ---- 6. GCD property gcd(F_m, F_n) = F_gcd(m,n) --------------------------------------
    ok = True
    for m in range(1, 30):
        for n in range(1, 30):
            if math.gcd(fibonacci(m), fibonacci(n)) != fibonacci(math.gcd(m, n)):
                ok = False
                break
        if not ok:
            break
    check("gcd(F_m,F_n) = F_gcd(m,n)", ok)

    # ---- 7. F_m | F_n iff m | n (strong divisibility) -----------------------------------
    # start at m=3: F_2 = 1 divides everything, so the "iff" degenerates only there.
    ok = True
    for m in range(3, 20):
        for n in range(m, 60):
            divides = (fibonacci(n) % fibonacci(m) == 0)
            if divides != (n % m == 0):
                ok = False
                break
        if not ok:
            break
    check("F_m | F_n iff m | n", ok)

    # ---- 8. fib_mod matches fibonacci mod m (incl huge n) -------------------------------
    ok = all(fib_mod(n, 1000) == fibonacci(n) % 1000 for n in range(500))
    check("fib_mod == fibonacci mod m (n=0..499)", ok)
    # a huge index, only feasible via modular fast doubling
    check("fib_mod(10^18, 1000003) computes", isinstance(fib_mod(10**18, 1000003), int))

    # ---- 9. Pisano period genuinely cycles F_n mod m ------------------------------------
    for m in [2, 3, 5, 10, 7]:
        p = pisano_period(m)
        # F_{n+p} == F_n mod m for a range of n
        ok = all(fib_mod(n + p, m) == fib_mod(n, m) for n in range(30))
        check(f"Pisano period pi({m})={p} cycles", ok)
    check("pi(10) = 60", pisano_period(10) == 60)  # classic: F mod 10 repeats every 60

    # ---- 10. Zeckendorf: non-consecutive Fibonacci sum ----------------------------------
    ok = True
    fibset = set(fibonacci(k) for k in range(2, 40))
    for n in range(1, 500):
        rep = zeckendorf(n)
        if sum(rep) != n:
            ok = False
            break
        # all terms are Fibonacci numbers
        if not all(f in fibset for f in rep):
            ok = False
            break
        # non-consecutive: no two chosen Fibonacci numbers are adjacent in the sequence
        idxs = sorted(fib_index(f) for f in rep)
        if any(idxs[i + 1] - idxs[i] < 2 for i in range(len(idxs) - 1)):
            ok = False
            break
    check("Zeckendorf: non-consecutive, sums to n (1..499)", ok)

    # ---- 11. fib_index inverse ----------------------------------------------------------
    check("fib_index(55) = 10", fib_index(55) == 10)
    check("fib_index(354224848179261915075) = 100", fib_index(354224848179261915075) == 100)
    check("fib_index(4) = None (not Fibonacci)", fib_index(4) is None)

    # ---- 12. negafibonacci F_{-n} = (-1)^{n+1} F_n --------------------------------------
    check("F_{-1}=1, F_{-2}=-1, F_{-6}=-8",
          fibonacci(-1) == 1 and fibonacci(-2) == -1 and fibonacci(-6) == -8)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
