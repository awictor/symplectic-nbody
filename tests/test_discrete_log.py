"""Tests for discrete_log: BSGS roundtrip, vs brute force, no-solution, DH recovery."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from discrete_log import discrete_log, brute_discrete_log, multiplicative_order

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


PRIMES = [p for p in range(5, 300) if is_prime(p)]

# --- the returned x satisfies g^x = h (mod m) ------------------------------
ok = True
for p in PRIMES[:25]:
    for g in [2, 3]:
        if g >= p:
            continue
        for h in range(1, min(p, 15)):
            x = discrete_log(g, h, p)
            if x is not None and pow(g, x, p) != h % p:
                ok = False
                break
        if not ok:
            break
    if not ok:
        break
check("BSGS: g^x = h (mod p) whenever a solution is returned", ok)

# --- matches brute force (smallest exponent, and existence) ----------------
ok = True
for p in PRIMES[:25]:
    for g in [2, 3, 5]:
        if g >= p:
            continue
        for h in range(1, p):
            bsgs = discrete_log(g, h, p)
            brute = brute_discrete_log(g, h, p)
            # existence must agree
            if (bsgs is None) != (brute is None):
                ok = False
                break
            # both must be valid solutions (BSGS need not return the literal smallest, but must be a
            # valid exponent; check validity, and that when brute exists BSGS does too)
            if bsgs is not None and pow(g, bsgs, p) != h % p:
                ok = False
                break
        if not ok:
            break
    if not ok:
        break
check("BSGS existence and validity agree with brute force", ok)

# --- known small case ------------------------------------------------------
x = discrete_log(3, 13, 17)
check("3^x = 13 mod 17 has x with 3^x = 13", x is not None and pow(3, x, 17) == 13)

# --- no-solution case ------------------------------------------------------
# 4 mod 7 has order 3, its powers are {1, 4, 2}; 3 is not among them
check("no discrete log for 4^x = 3 mod 7", discrete_log(4, 3, 7, order=3) is None)
# 2 mod 7 has order 3 as well (2,4,1); 5 is not a power of 2
check("no discrete log for 2^x = 5 mod 7", brute_discrete_log(2, 5, 7) is None)

# --- multiplicative order --------------------------------------------------
check("order of 3 mod 17 is 16 (generator)", multiplicative_order(3, 17) == 16)
check("order of 2 mod 7 is 3", multiplicative_order(2, 7) == 3)
check("order of 1 is 1", multiplicative_order(1, 13) == 1)
check("non-invertible element has no order", multiplicative_order(4, 8) is None)

# --- Diffie-Hellman style recovery -----------------------------------------
# public g, p; secret exponent a; public value A = g^a; recover a from A
p, g = 467, 2
for secret in [7, 100, 200, 300]:
    A = pow(g, secret, p)
    recovered = discrete_log(g, A, p)
    check(f"DH: recovered exponent reproduces the public value (secret {secret})",
          recovered is not None and pow(g, recovered, p) == A)

# --- x = 0 case (h = 1) ----------------------------------------------------
check("g^0 = 1: discrete_log(g, 1, p) is 0", discrete_log(5, 1, 101) == 0)

# --- larger prime ----------------------------------------------------------
p = 100003
g = 2
secret = 54321
A = pow(g, secret, p)
x = discrete_log(g, A, p)
check("larger prime: recovered exponent reproduces the value",
      x is not None and pow(g, x, p) == A)

# --- BSGS is faster than brute (structural: same answers, sqrt work) -------
# just confirm agreement on a medium prime for several targets
p = 7919
g = 7
ok = True
for h in [pow(g, e, p) for e in [10, 500, 3000, 7000]]:
    x = discrete_log(g, h, p)
    if x is None or pow(g, x, p) != h:
        ok = False
        break
check("medium prime: all targets solved correctly", ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all discrete_log tests passed")
