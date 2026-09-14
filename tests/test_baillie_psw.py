"""Tests for Baillie-PSW: agreement with trial division, pseudoprime catching, Jacobi symbol, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
    # ---- 1. agrees with trial division for every n up to a bound ------------------------
    mism = [n for n in range(0, 20000) if B.is_prime(n) != B.trial_division_prime(n)]
    check("Baillie-PSW == trial division for 0..19999", not mism, f"first mismatches {mism[:5]}")

    # ---- 2. small primes recognized -----------------------------------------------------
    check("recognizes 2, 3, 5, 7", all(B.is_prime(p) for p in (2, 3, 5, 7)))
    check("rejects 0, 1", not B.is_prime(0) and not B.is_prime(1))
    check("rejects small composites", not any(B.is_prime(c) for c in (4, 6, 8, 9, 15, 21, 25)))

    # ---- 3. strong pseudoprimes to base 2 are caught by the Lucas half ------------------
    # these fool a naive Fermat/Miller-Rabin base-2 test but are composite
    base2_psp = [2047, 3277, 4033, 4681, 8321, 15841, 29341]
    check("strong base-2 pseudoprimes rejected",
          not any(B.is_prime(x) for x in base2_psp),
          f"{[x for x in base2_psp if B.is_prime(x)]}")

    # ---- 4. Carmichael numbers (fool the plain Fermat test) are rejected ----------------
    carmichael = [561, 1105, 1729, 2465, 2821, 6601, 8911, 10585]
    check("Carmichael numbers rejected", not any(B.is_prime(c) for c in carmichael),
          f"{[c for c in carmichael if B.is_prime(c)]}")

    # ---- 5. Mersenne and other large primes ---------------------------------------------
    check("2^31 - 1 is prime", B.is_prime(2 ** 31 - 1))
    check("10^9 + 7 is prime", B.is_prime(10 ** 9 + 7))
    check("2^31 is composite", not B.is_prime(2 ** 31))
    check("a large composite (product of two primes) rejected",
          not B.is_prime(1000003 * 1000033))

    # ---- 6. Jacobi symbol matches the Legendre symbol for an odd prime ------------------
    p = 41
    ok = True
    for a in range(1, p):
        leg = pow(a, (p - 1) // 2, p)
        leg = 1 if leg == 1 else -1
        if B.jacobi_symbol(a, p) != leg:
            ok = False
    check("Jacobi == Legendre for prime 41", ok)

    # ---- 7. Jacobi symbol is multiplicative ---------------------------------------------
    n = 15
    mult_ok = all(B.jacobi_symbol((a * b) % n, n) == B.jacobi_symbol(a, n) * B.jacobi_symbol(b, n)
                  for a in range(1, n) for b in range(1, n))
    check("Jacobi symbol is multiplicative (n=15)", mult_ok)
    check("Jacobi (a/n) = 0 when gcd(a,n)>1", B.jacobi_symbol(6, 15) == 0)

    # ---- 8. perfect squares are composite (except handled specially) --------------------
    check("perfect squares rejected", not any(B.is_prime(k * k) for k in range(2, 50)))
    check("is_perfect_square works", B.is_perfect_square(49) and not B.is_perfect_square(50))

    # ---- 9. the strong Miller-Rabin base-2 test alone accepts its pseudoprimes ----------
    # (demonstrates WHY the Lucas half is needed: MR base-2 alone would call 2047 prime)
    check("MR base-2 alone is fooled by 2047", B.strong_miller_rabin_base2(2047))
    check("but Baillie-PSW rejects 2047", not B.is_prime(2047))

    # ---- 10. count of primes below 10000 matches the known value (1229) -----------------
    count = sum(1 for n in range(2, 10000) if B.is_prime(n))
    check("pi(10000) == 1229", count == 1229, f"{count}")

    # ---- 11. twin primes and a big prime gap ------------------------------------------
    check("recognizes twin primes 10007, 10009", B.is_prime(10007) and B.is_prime(10009))
    check("larger prime 999999937", B.is_prime(999999937))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
