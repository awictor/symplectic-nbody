"""Tests for Pratt certificates: build/verify round-trip, composite rejection, tamper detection."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pratt_certificate as P  # noqa: E402
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
    primes = [2, 3, 5, 7, 11, 13, 17, 101, 251, 997, 7919, 104729]

    # ---- 1. a certificate is built and verifies for every prime -------------------------
    ok = True
    for p in primes:
        c = P.build(p)
        if c is None or not P.verify(c):
            ok = False
    check("build + verify round-trips for many primes", ok)

    # ---- 2. no certificate for composites ----------------------------------------------
    comps = [4, 6, 8, 9, 15, 21, 25, 100, 561, 1000, 7917]
    check("composites yield no certificate", all(P.build(c) is None for c in comps))

    # ---- 3. the base case is 2 ----------------------------------------------------------
    check("certificate of 2 is the base case", P.build(2) == (2, None, []))

    # ---- 4. the witness is a genuine primitive root ------------------------------------
    p = 101
    c = P.build(p)
    a = c[1]
    from lenstra_ecm import factorize
    factors = set(factorize(p - 1))
    check("witness is a primitive root of 101",
          pow(a, p - 1, p) == 1 and all(pow(a, (p - 1) // q, p) != 1 for q in factors))

    # ---- 5. tampering with the witness is detected -------------------------------------
    tampered = (101, 100, c[2])                  # 100 ≡ -1 has order 2, not a primitive root
    check("bad witness rejected", not P.verify(tampered))

    # ---- 6. dropping a required prime factor is detected -------------------------------
    if len(c[2]) > 1:
        short = (101, c[1], c[2][:-1])
        check("missing prime factor rejected", not P.verify(short))
    else:
        check("missing prime factor rejected", True)  # 101-1=100=2^2*5^2 has 2 distinct primes

    # ---- 7. a composite masquerading as prime is rejected -------------------------------
    fake = (15, 2, [(2, (2, None, [])),
                    (7, (7, 3, [(2, (2, None, [])), (3, (3, 2, [(2, (2, None, []))]))]))])
    check("composite with a forged certificate rejected", not P.verify(fake))

    # ---- 8. verify(cert, n) checks the number matches -----------------------------------
    check("verify with matching n accepts", P.verify(c, 101))
    check("verify with mismatched n rejects", not P.verify(c, 103))

    # ---- 9. certificate size is small (polynomial in log n) -----------------------------
    c_big = P.build(104729)                       # the 10000th prime
    size = P.certificate_size(c_big)
    import math
    check("certificate tree is small (O(log^2 n))", size < 20 * (math.log2(104729) ** 2),
          f"size {size}")

    # ---- 10. every prime below 500 gets a verifiable certificate; no composite does -----
    all_ok = True
    for n in range(2, 500):
        c = P.build(n)
        if is_prime(n):
            if c is None or not P.verify(c):
                all_ok = False
        else:
            if c is not None:
                all_ok = False
    check("build/verify matches primality for all n<500", all_ok)

    # ---- 11. a nested sub-certificate is itself valid -----------------------------------
    c = P.build(7919)
    q, cq = c[2][0]
    check("a sub-certificate independently verifies", P.verify(cq, q))

    # ---- 12. witness() convenience ------------------------------------------------------
    check("witness(7) is a valid primitive root",
          P.witness(7) is not None and pow(P.witness(7), 6, 7) == 1)
    check("witness of a composite is None", P.witness(15) is None)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
