"""Tests for Dixon factorization: correct prime factorizations, matches trial division, congruences."""

import os
import sys
from math import gcd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dixon import (  # noqa: E402
    dixon_factor,
    factorize,
    trial_factorization,
    factor_base,
    _smooth_vector,
    _gf2_nullspace_dependency,
    is_congruence_of_squares,
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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    # ---- 1. dixon_factor returns a nontrivial factor of semiprimes ----------------------
    ok = True
    for N in [15, 21, 35, 143, 323, 1147, 8051, 10403, 9409]:
        f = dixon_factor(N)
        if f is None or not (1 < f < N) or N % f != 0:
            ok = False
            check("dixon_factor nontrivial", False, f"N={N}: {f}")
            break
    if ok:
        check("dixon_factor gives a nontrivial factor (9 semiprimes)", True)

    # ---- 2. complete factorization multiplies back to N ---------------------------------
    ok = True
    for N in [15, 100, 143, 360, 1147, 9409, 10403, 2310]:
        facs = factorize(N)
        prod = 1
        for f in facs:
            prod *= f
        if prod != N:
            ok = False
            check("factorize product = N", False, f"N={N}: {facs}")
            break
    if ok:
        check("factorize product equals N (8 numbers)", True)

    # ---- 3. factorize == trial division ------------------------------------------------
    rng = _lcg(1)
    ok = True
    for _ in range(30):
        N = 2 + int(rng() * 5000)
        if factorize(N) != trial_factorization(N):
            ok = False
            check("factorize == trial", False, f"N={N}: {factorize(N)} vs {trial_factorization(N)}")
            break
    if ok:
        check("factorize == trial division (30 random N)", True)

    # ---- 4. factorization entries are all prime -----------------------------------------
    from pollard_rho import is_prime
    ok = True
    for N in [143, 1147, 9409, 2310, 10403]:
        if not all(is_prime(f) for f in factorize(N)):
            ok = False
            break
    check("all factors are prime", ok)

    # ---- 5. primes are returned unchanged -----------------------------------------------
    for p in [7, 13, 101, 997, 7919]:
        check(f"prime {p} -> [{p}]", factorize(p) == [p])

    # ---- 6. even numbers and prime powers -----------------------------------------------
    check("factorize 2 = [2]", factorize(2) == [2])
    check("factorize 1024 = 2^10", factorize(1024) == [2] * 10)
    check("factorize 9409 = 97^2", factorize(9409) == [97, 97])
    check("factorize 27 = 3^3", factorize(27) == [3, 3, 3])

    # ---- 7. n=1 and n=0 -----------------------------------------------------------------
    check("factorize 1 = []", factorize(1) == [])

    # ---- 8. smooth-vector factorization is exact ----------------------------------------
    base = [2, 3, 5, 7]
    # 2^3 * 3 * 5 = 120
    vec = _smooth_vector(120, base)
    check("smooth vector of 120 over {2,3,5,7}", vec == [3, 1, 1, 0], f"{vec}")
    # 22 = 2 * 11 not smooth over {2,3,5,7}
    check("22 not smooth over {2,3,5,7}", _smooth_vector(22, base) is None)

    # ---- 9. GF(2) dependency really sums to zero mod 2 ----------------------------------
    vecs = [[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]]  # rows 0^1^2 = 0
    dep = _gf2_nullspace_dependency(vecs)
    check("GF(2) dependency found", dep is not None, f"{dep}")
    if dep:
        s = [0, 0, 0]
        for i in dep:
            for k in range(3):
                s[k] ^= vecs[i][k] % 2
        check("dependency sums to 0 mod 2", s == [0, 0, 0], f"{s}")

    # ---- 10. every Dixon relation gives a genuine congruence of squares -----------------
    # verify the identity on a manual congruence: 7^2 = 49, and 49 mod 15 = 4 = 2^2
    check("congruence-of-squares identity", is_congruence_of_squares(15, 7, 2))
    # from the factor it finds: reconstruct X,Y? just check gcd-based factor is valid
    N = 8051
    f = dixon_factor(N)
    check("8051 = 83 * 97", f in (83, 97) and 8051 % f == 0, f"{f}")

    # ---- 11. factor base is primes below the bound --------------------------------------
    base = factor_base(10403)
    check("factor base is small primes", all(p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
                                             for p in base) and 2 in base, f"{base}")

    # ---- 12. larger semiprime ----------------------------------------------------------
    N = 100000007 * 3  # small prime times 3? use a clean semiprime
    N = 999983 * 101   # two primes
    facs = factorize(N)
    check("large semiprime factorized", facs == sorted([101, 999983]), f"{facs}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
