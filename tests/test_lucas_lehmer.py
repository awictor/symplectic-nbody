"""Tests for Lucas-Lehmer: Mersenne primes correct, agrees with primality, Lucas sequences + identity."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lucas_lehmer import (  # noqa: E402
    lucas_lehmer,
    mersenne_prime_exponents,
    lucas_sequence,
    lucas_sequence_naive,
    is_lucas_probable_prime,
    mersenne_reduction_check,
    _mersenne_mod,
    D_of,
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
    # ---- 1. known Mersenne-prime exponents ----------------------------------------------
    # M_p prime for p in {2,3,5,7,13,17,19,31,61,89,107,127}
    known = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}
    got = set(mersenne_prime_exponents(130))
    check("Mersenne prime exponents up to 130", got == known, f"{sorted(got)}")

    # ---- 2. composite Mersenne exponents rejected --------------------------------------
    for p in [11, 23, 29, 37, 41, 43, 47, 53, 59, 67, 71, 73, 79, 83, 97, 101, 103, 109, 113]:
        check(f"M_{p} composite (rejected)", not lucas_lehmer(p)) if p in (11, 23, 67) else None
    # spot-check a few
    check("M_11 = 2047 = 23*89 composite", not lucas_lehmer(11))
    check("M_23 composite", not lucas_lehmer(23))
    check("M_67 composite (Cole's famous factorization)", not lucas_lehmer(67))

    # ---- 3. Lucas-Lehmer agrees with a real primality check on 2^p - 1 ------------------
    # cap at p<=62: pollard-rho is_prime bogs down factoring the huge composite Mersennes above that
    ok = True
    for p in range(2, 63):
        ll = lucas_lehmer(p)
        actual = is_prime((1 << p) - 1)
        if ll != actual:
            ok = False
            check("LL == is_prime(2^p-1)", False, f"p={p}: LL={ll} actual={actual}")
            break
    if ok:
        check("Lucas-Lehmer == primality of 2^p-1 (p=2..62)", True)

    # ---- 4. base cases ------------------------------------------------------------------
    check("M_2 = 3 prime", lucas_lehmer(2))
    check("M_3 = 7 prime", lucas_lehmer(3))
    check("M_5 = 31 prime", lucas_lehmer(5))

    # ---- 5. Mersenne fast reduction == ordinary modulo ---------------------------------
    ok = True
    for p in [3, 5, 7, 13, 31]:
        for x in [0, 1, (1 << p) - 1, (1 << p), (1 << (2 * p)) - 5, 12345678901234567890]:
            if not mersenne_reduction_check(x, p):
                ok = False
                break
        if not ok:
            break
    check("Mersenne fast reduction == modulo", ok)

    # ---- 6. Lucas sequences fast doubling == naive recurrence ---------------------------
    ok = True
    for P, Q in [(1, -1), (2, -1), (1, -2), (3, 2)]:
        for n in range(0, 60):
            fast = lucas_sequence(n, P, Q)[:2]
            naive = lucas_sequence_naive(n, P, Q)
            if fast != naive:
                ok = False
                check("Lucas fast == naive", False, f"P={P} Q={Q} n={n}: {fast} vs {naive}")
                break
        if not ok:
            break
    if ok:
        check("Lucas sequence fast doubling == naive (4 params)", True)

    # ---- 7. (P,Q)=(1,-1) reproduces Fibonacci (U) and Lucas (V) numbers ------------------
    # U_n(1,-1) = F_n, V_n(1,-1) = L_n
    from fibonacci import fibonacci, lucas as lucas_num
    ok = all(lucas_sequence(n, 1, -1)[0] == fibonacci(n) for n in range(30))
    check("U_n(1,-1) = Fibonacci", ok)
    ok = all(lucas_sequence(n, 1, -1)[1] == lucas_num(n) for n in range(30))
    check("V_n(1,-1) = Lucas numbers", ok)

    # ---- 8. identity V_n^2 - D U_n^2 = 4 Q^n --------------------------------------------
    ok = True
    for P, Q in [(1, -1), (3, 2), (2, -1)]:
        D = D_of(P, Q)
        for n in range(0, 40):
            U, V, qn = lucas_sequence(n, P, Q)
            if V * V - D * U * U != 4 * qn:
                ok = False
                break
        if not ok:
            break
    check("identity V_n^2 - D U_n^2 = 4 Q^n", ok)

    # ---- 9. modular Lucas sequence matches unreduced mod m ------------------------------
    ok = True
    for n in range(0, 40):
        Um, Vm, _ = lucas_sequence(n, 1, -1, m=97)
        U, V, _ = lucas_sequence(n, 1, -1)
        if Um != U % 97 or Vm != V % 97:
            ok = False
            break
    check("modular Lucas == unreduced mod m", ok)

    # ---- 10. Lucas probable-prime test flags primes -------------------------------------
    # primes should pass the Lucas PRP test (P,Q)=(1,-1)
    ok = all(is_lucas_probable_prime(p) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
    check("primes are Lucas PRPs", ok)
    # obvious composites (that are not Lucas pseudoprimes) fail
    check("9 is not a Lucas PRP", not is_lucas_probable_prime(9))
    check("15 is not a Lucas PRP", not is_lucas_probable_prime(15))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
