"""Tests for crt: CRT solution vs brute force, extended Euclid, general non-coprime CRT."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crt import extended_gcd, mod_inverse, crt, crt_general, verify

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 191
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- the classic Sunzi puzzle ----------------------------------------------
x, M = crt([2, 3, 2], [3, 5, 7])
check("Sunzi puzzle solution is 23 mod 105", x == 23 and M == 105)
check("Sunzi solution verifies", verify(x, [2, 3, 2], [3, 5, 7]))

# --- extended Euclid produces a correct Bezout identity --------------------
ok = True
for _ in range(300):
    a = 1 + int(rng() * 1000)
    b = 1 + int(rng() * 1000)
    g, x, y = extended_gcd(a, b)
    if g != math.gcd(a, b) or a * x + b * y != g:
        ok = False
        break
check("extended_gcd gives gcd and a valid Bezout identity", ok)

# --- modular inverse -------------------------------------------------------
ok = True
for _ in range(300):
    m = 2 + int(rng() * 1000)
    a = 1 + int(rng() * (m - 1))
    if math.gcd(a, m) != 1:
        continue
    inv = mod_inverse(a, m)
    if (a * inv) % m != 1 or not (0 <= inv < m):
        ok = False
        break
check("mod_inverse gives a correct inverse in [0, m)", ok)
# non-coprime raises
raised = False
try:
    mod_inverse(4, 8)
except ValueError:
    raised = True
check("mod_inverse of a non-coprime raises", raised)

# --- CRT solution satisfies every congruence and matches brute force -------
ok = True
for _ in range(200):
    # pick 2-4 pairwise-coprime moduli from a small set of primes
    primes = [2, 3, 5, 7, 11, 13]
    k = 2 + int(rng() * 3)
    # sample k distinct primes
    idx = sorted(range(len(primes)), key=lambda i: (rng(), i))[:k]
    moduli = [primes[i] for i in idx]
    remainders = [int(rng() * m) for m in moduli]
    x, M = crt(remainders, moduli)
    if not verify(x, remainders, moduli):
        ok = False
        break
    if not (0 <= x < M):
        ok = False
        break
    # brute force: the smallest non-negative solution
    brute = next(v for v in range(M) if verify(v, remainders, moduli))
    if x != brute:
        ok = False
        break
check("CRT matches brute force over 200 coprime systems", ok)

# --- CRT with two moduli ---------------------------------------------------
x, M = crt([1, 2], [4, 5])   # x ≡ 1 mod 4, x ≡ 2 mod 5 -> 17
check("two-modulus CRT is 17 mod 20", x == 17 and M == 20)

# --- generalized CRT: non-coprime but consistent ---------------------------
# x ≡ 2 mod 6, x ≡ 8 mod 12 : both satisfied by 8 mod 12
res = crt_general([2, 8], [6, 12])
check("consistent non-coprime system solved", res is not None and verify(res[0], [2, 8], [6, 12]))

# x ≡ 3 mod 4, x ≡ 5 mod 6 : gcd(4,6)=2, 3≡1 and 5≡1 mod 2 consistent -> 11 mod 12
res = crt_general([3, 5], [4, 6])
check("non-coprime consistent gives 11 mod 12", res == (11, 12))

# --- generalized CRT: contradictory ----------------------------------------
check("contradictory system (1 mod 2, 0 mod 4) has no solution",
      crt_general([1, 0], [2, 4]) is None)
check("contradictory system (2 mod 4, 1 mod 6) rejected",
      crt_general([2, 1], [4, 6]) is None)

# --- generalized CRT matches ordinary CRT on coprime moduli ----------------
ok = True
for _ in range(100):
    moduli = [3, 5, 7]
    remainders = [int(rng() * m) for m in moduli]
    x1, M1 = crt(remainders, moduli)
    res = crt_general(remainders, moduli)
    if res != (x1, M1):
        ok = False
        break
check("general CRT agrees with coprime CRT", ok)

# --- generalized CRT solution is the smallest non-negative -----------------
res = crt_general([2, 8], [6, 12])
check("general CRT returns the smallest non-negative solution",
      res is not None and 0 <= res[0] < res[1])

# --- a single congruence ---------------------------------------------------
check("single congruence x = 3 mod 7 -> 3", crt([3], [7]) == (3, 7))

# --- large coprime moduli --------------------------------------------------
x, M = crt([12345, 6789], [1000003, 1000033])
check("large coprime CRT verifies", verify(x, [12345, 6789], [1000003, 1000033]) and 0 <= x < M)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all crt tests passed")
