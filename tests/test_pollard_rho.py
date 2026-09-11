"""Tests for pollard_rho: factorization vs brute force, primality vs sieve, semiprimes, totient."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pollard_rho import (is_prime, pollard_rho, pollard_p_minus_1, factorize,
                         factor_pairs, euler_phi, num_divisors)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 13
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- primality vs a sieve --------------------------------------------------
def sieve(limit):
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
    return is_p


N = 5000
is_p = sieve(N)
prim_ok = all(is_prime(n) == is_p[n] for n in range(2, N + 1))
check("Miller-Rabin matches a sieve up to 5000", prim_ok)

# some known large primes and composites
check("2^31 - 1 is prime (Mersenne)", is_prime(2147483647))
check("2^31 is not prime", not is_prime(2147483648))
check("a big prime is prime", is_prime(1000000007))
check("a big semiprime is not prime", not is_prime(1000000007 * 1000000009))

# --- factorization: product equals input, all factors prime ----------------
def check_factorization(n):
    fs = factorize(n)
    prod = 1
    for f in fs:
        prod *= f
    return prod == n and all(is_prime(f) for f in fs)


ok = True
for n in [2, 3, 4, 12, 100, 1000, 13195, 999983, 1000000, 123456789, 600851475143]:
    if not check_factorization(n):
        ok = False
        break
check("factorization product == input and all factors prime (known values)", ok)

# --- factorization matches brute-force trial division ----------------------
def brute_factorize(n):
    out = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


ok = True
for _ in range(200):
    n = 2 + int(rng() * 100000)
    if factorize(n) != brute_factorize(n):
        ok = False
        break
check("factorization matches brute-force trial division over 200 random n", ok)

# --- semiprimes (the RSA case): recover both prime factors -----------------
primes = [p for p in range(1000, 2000) if is_prime(p)]
ok = True
for _ in range(30):
    p = primes[int(rng() * len(primes))]
    q = primes[int(rng() * len(primes))]
    n = p * q
    if sorted(factorize(n)) != sorted([p, q]):
        ok = False
        break
check("recovers both factors of random semiprimes", ok)

# --- a larger semiprime ----------------------------------------------------
p, q = 1000003, 1000033
check("factors a ~12-digit semiprime", sorted(factorize(p * q)) == [p, q])

# --- pollard_rho returns None on primes, a real factor on composites -------
check("pollard_rho on a prime returns None", pollard_rho(1000000007) is None)
f = pollard_rho(1000003 * 1000033)
check("pollard_rho finds a non-trivial factor", f is not None and (p * q) % f == 0 and 1 < f < p * q)

# --- Pollard p-1 on a smooth-prime composite -------------------------------
# 10403 = 101 * 103; 101-1 = 100 = 2^2*5^2 is smooth
d = pollard_p_minus_1(10403)
check("pollard p-1 finds a factor of a smooth semiprime", d is not None and 10403 % d == 0 and 1 < d < 10403)

# --- factor_pairs (prime, exponent) ---------------------------------------
check("factor_pairs of 1000 is 2^3 * 5^3", factor_pairs(1000) == [(2, 3), (5, 3)])
check("factor_pairs of a prime", factor_pairs(97) == [(97, 1)])
check("factor_pairs of 1 is empty", factor_pairs(1) == [])

# --- perfect powers and highly composite numbers ---------------------------
check("2^20 factors correctly", factorize(2 ** 20) == [2] * 20)
check("factorial 15 factors correctly", check_factorization(1307674368000))  # 15!
check("720720 (highly composite) factors correctly", check_factorization(720720))

# --- Euler totient vs brute-force count ------------------------------------
def brute_phi(n):
    from math import gcd
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)


ok = all(euler_phi(n) == brute_phi(n) for n in range(1, 300))
check("euler_phi matches a brute-force coprime count up to 300", ok)

# --- number of divisors vs brute count -------------------------------------
def brute_divisors(n):
    return sum(1 for d in range(1, n + 1) if n % d == 0)


ok = all(num_divisors(n) == brute_divisors(n) for n in range(1, 300))
check("num_divisors matches a brute-force divisor count up to 300", ok)

# --- edge cases ------------------------------------------------------------
check("factorize(1) is empty", factorize(1) == [])
check("factorize(0) is empty", factorize(0) == [])
check("factorize of a prime is itself", factorize(7919) == [7919])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all pollard_rho tests passed")
