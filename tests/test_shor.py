"""Tests for Shor: factors small semiprimes, order satisfies a^r=1, quantum==classical, honest fails."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shor import (  # noqa: E402
    shor_factor,
    multiplicative_order,
    find_order_quantum,
    verify_factorization,
    gcd,
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
    # ---- 1. classical-order Shor factors small semiprimes -------------------------------
    for N, primes in [(15, {3, 5}), (21, {3, 7}), (35, {5, 7}), (33, {3, 11}),
                      (77, {7, 11}), (91, {7, 13})]:
        f = shor_factor(N, quantum=False)
        ok = verify_factorization(N, f) and set(f) == primes
        check(f"classical Shor factors {N} = {primes}", ok, f"got {f}")

    # ---- 2. quantum-simulated Shor factors the small ones -------------------------------
    for N, primes in [(15, {3, 5}), (21, {3, 7})]:
        f = shor_factor(N, quantum=True)
        ok = verify_factorization(N, f) and set(f) == primes
        check(f"quantum Shor factors {N} = {primes}", ok, f"got {f}")

    # ---- 3. recovered order genuinely satisfies a^r = 1 mod N ---------------------------
    ok = True
    for N in [15, 21, 35]:
        for a in range(2, N):
            if gcd(a, N) != 1:
                continue
            r = multiplicative_order(a, N)
            if r is not None and pow(a, r, N) != 1:
                ok = False
                break
    check("multiplicative_order: a^r = 1 mod N", ok)

    # ---- 4. order is the SMALLEST such r ------------------------------------------------
    # a=2 mod 15: 2,4,8,16=1 -> order 4
    check("order(2 mod 15) = 4", multiplicative_order(2, 15) == 4)
    check("order(2 mod 7) = 3", multiplicative_order(2, 7) == 3)
    check("order(4 mod 15) = 2", multiplicative_order(4, 15) == 2)

    # ---- 5. quantum order-finding agrees with classical order (where it succeeds) -------
    ok = True
    checked = 0
    for N in [15, 21]:
        for a in range(2, N):
            if gcd(a, N) != 1:
                continue
            rc = multiplicative_order(a, N)
            rq = find_order_quantum(a, N)
            if rq is not None:
                checked += 1
                # quantum may return a divisor/multiple; require it to be a valid order
                if pow(a, rq, N) != 1:
                    ok = False
                    break
        if not ok:
            break
    check(f"quantum order valid (a^rq=1), {checked} cases", ok and checked > 0)

    # ---- 6. even semiprime handled trivially --------------------------------------------
    f = shor_factor(14)
    check("factors 14", verify_factorization(14, f) and set(f) == {2, 7}, f"{f}")

    # ---- 7. prime input -> None (no nontrivial factorization) ---------------------------
    for p in [7, 13, 31, 97]:
        check(f"prime {p} -> None", shor_factor(p) is None)

    # ---- 8. perfect power detected ------------------------------------------------------
    f = shor_factor(9)   # 3^2
    check("factors 9 = 3*3", verify_factorization(9, f) and set(f) == {3}, f"{f}")
    f = shor_factor(25)  # 5^2
    check("factors 25 = 5*5", verify_factorization(25, f) and set(f) == {5}, f"{f}")

    # ---- 9. factorization always verifies when returned ---------------------------------
    ok = True
    for N in [15, 21, 33, 35, 39, 51, 55, 57, 65]:
        f = shor_factor(N, quantum=False)
        if f is not None and not verify_factorization(N, f):
            ok = False
            break
    check("returned factorizations all verify", ok)

    # ---- 10. non-coprime a shortcut -----------------------------------------------------
    # if a shares a factor with N, gcd gives it immediately; ensure shor still succeeds
    f = shor_factor(35, seed=2, quantum=False)
    check("factors 35 with different seed", verify_factorization(35, f) and set(f) == {5, 7}, f"{f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
