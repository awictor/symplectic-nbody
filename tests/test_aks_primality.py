"""Tests for AKS primality: agreement with trial division and Baillie-PSW, poly identity, perfect powers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import aks_primality as A  # noqa: E402
import baillie_psw as B  # noqa: E402


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
    # ---- 1. agrees with deterministic trial division for every n in a range ------------
    mism = [n for n in range(2, 600) if A.is_prime(n) != A.trial_division_prime(n)]
    check("AKS == trial division for 2..599", not mism, f"first mismatches {mism[:5]}")

    # ---- 2. agrees with Baillie-PSW ----------------------------------------------------
    mism2 = [n for n in range(2, 800) if A.is_prime(n) != B.is_prime(n)]
    check("AKS == Baillie-PSW for 2..799", not mism2, f"{mism2[:5]}")

    # ---- 3. small primes and composites ------------------------------------------------
    check("recognizes 2, 3, 5, 7, 11, 13", all(A.is_prime(p) for p in (2, 3, 5, 7, 11, 13)))
    check("rejects 0, 1, 4, 6, 8, 9", not any(A.is_prime(c) for c in (0, 1, 4, 6, 8, 9)))

    # ---- 4. Carmichael numbers and strong pseudoprimes are correctly composite ---------
    for c in (341, 561, 645, 1105, 1729, 2047):
        check(f"{c} is composite (AKS)", not A.is_prime(c))

    # ---- 5. perfect powers detected -----------------------------------------------------
    check("is_perfect_power detects 27, 1024, 3^7", A.is_perfect_power(27) and
          A.is_perfect_power(1024) and A.is_perfect_power(3 ** 7))
    check("is_perfect_power rejects primes and non-powers",
          not A.is_perfect_power(7) and not A.is_perfect_power(12))
    check("AKS rejects perfect powers as composite",
          not A.is_prime(49) and not A.is_prime(125))

    # ---- 6. the polynomial identity holds for primes, fails for composites -------------
    check("(x+a)^7 == x^7 + a mod (x^5-1, 7) for prime 7", A._check_polynomial(7, 5, 1))
    check("(x+a)^9 != x^9 + a for composite 9", not A._check_polynomial(9, 5, 1))
    check("(x+a)^11 holds for prime 11", A._check_polynomial(11, 7, 2))
    check("(x+a)^15 fails for composite 15", not A._check_polynomial(15, 7, 1))

    # ---- 7. multiplicative order subroutine --------------------------------------------
    check("ord_7(2) == 3 (2^3=8=1 mod 7)", A._multiplicative_order(2, 7) == 3)
    check("ord_r undefined when gcd != 1", A._multiplicative_order(6, 9) is None)

    # ---- 8. Euler phi subroutine --------------------------------------------------------
    check("phi(10) == 4", A._euler_phi(10) == 4)
    check("phi(p) == p-1 for prime", A._euler_phi(13) == 12)
    check("phi(12) == 4", A._euler_phi(12) == 4)

    # ---- 9. certificate returns sensible parameters ------------------------------------
    ok, r, ab = A.is_prime(31, certificate=True)
    check("certificate for 31: prime with r and a_bound", ok and r is not None and ab is not None,
          f"{(ok, r, ab)}")
    okc, rc, abc = A.is_prime(33, certificate=True)
    check("certificate for 33: composite", not okc)

    # ---- 10. some larger primes ---------------------------------------------------------
    check("997 is prime", A.is_prime(997))
    check("1009 is prime", A.is_prime(1009))
    check("1000 is composite", not A.is_prime(1000))

    # ---- 11. count of primes below 500 matches known (95) ------------------------------
    count = sum(1 for n in range(2, 500) if A.is_prime(n))
    check("pi(500) == 95", count == 95, f"{count}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
