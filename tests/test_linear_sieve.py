"""Tests for linear_sieve: primes, SPF, totient, Mobius vs brute force across the full range."""

import os
import sys
from math import gcd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linear_sieve import (linear_sieve, primes_up_to, factorize_with_spf, is_prime_sieve,
                          brute_is_prime, brute_factorize, brute_totient, brute_mobius)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


N = 3000
S = linear_sieve(N)

# --- primes -----------------------------------------------------------------
check("first primes are 2,3,5,7,11,13", S["primes"][:6] == [2, 3, 5, 7, 11, 13])
check("primes_up_to(30) is correct",
      primes_up_to(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
brute_primes = [n for n in range(2, N + 1) if brute_is_prime(n)]
check(f"sieve primes match trial division up to {N}", S["primes"] == brute_primes)
check("prime count up to 3000 is 430", len(S["primes"]) == 430)

# --- primality via SPF ------------------------------------------------------
prime_ok = all(is_prime_sieve(n, S["spf"]) == brute_is_prime(n) for n in range(2, N + 1))
check(f"is_prime_sieve matches trial division up to {N}", prime_ok)

# --- smallest prime factor / factorization ---------------------------------
fact_ok = True
for n in range(2, N + 1):
    if factorize_with_spf(n, S["spf"]) != brute_factorize(n):
        fact_ok = False
        print(f"  factorization mismatch at {n}")
        break
check(f"SPF factorization matches trial division up to {N}", fact_ok)

# every spf entry really is the smallest prime factor
spf_ok = True
for n in range(2, N + 1):
    p = S["spf"][n]
    # p divides n, p is prime, and no smaller prime divides n
    if n % p != 0 or not brute_is_prime(p):
        spf_ok = False
        break
    if any(n % q == 0 for q in range(2, p)):
        spf_ok = False
        break
check("every SPF entry is genuinely the smallest prime factor", spf_ok)

# --- Euler totient ----------------------------------------------------------
phi_ok = all(S["phi"][n] == brute_totient(n) for n in range(1, N + 1))
check(f"totient array matches the coprime-counting definition up to {N}", phi_ok)
check("phi(1)=1, phi(prime p)=p-1", S["phi"][1] == 1 and S["phi"][7] == 6 and S["phi"][13] == 12)
check("phi(12)=4, phi(36)=12", S["phi"][12] == 4 and S["phi"][36] == 12)

# sum of phi(d) over divisors d of n equals n (a classic identity), spot-checked
def divisor_sum_phi(n):
    return sum(S["phi"][d] for d in range(1, n + 1) if n % d == 0)


check("sum of phi over divisors of n equals n (identity)",
      all(divisor_sum_phi(n) == n for n in range(1, 200)))

# --- Mobius function --------------------------------------------------------
mu_ok = all(S["mu"][n] == brute_mobius(n) for n in range(1, N + 1))
check(f"Mobius array matches the squarefree/prime-count definition up to {N}", mu_ok)
check("mu(1)=1, mu(prime)=-1, mu(4)=0, mu(30)=-1",
      S["mu"][1] == 1 and S["mu"][2] == -1 and S["mu"][4] == 0 and S["mu"][30] == -1)

# sum of mu(d) over divisors of n is [n == 1] (the fundamental Mobius identity)
def divisor_sum_mu(n):
    return sum(S["mu"][d] for d in range(1, n + 1) if n % d == 0)


check("sum of mu over divisors of n is 1 if n==1 else 0 (Mobius identity)",
      all(divisor_sum_mu(n) == (1 if n == 1 else 0) for n in range(1, 300)))

# --- multiplicativity spot check -------------------------------------------
# phi(mn) = phi(m)phi(n) when gcd(m,n)=1
mult_ok = True
for m in range(1, 55):
    for n in range(1, 55):
        if gcd(m, n) == 1 and m * n <= N:
            if S["phi"][m * n] != S["phi"][m] * S["phi"][n]:
                mult_ok = False
                break
    if not mult_ok:
        break
check("phi is multiplicative on coprime arguments", mult_ok)

# --- edge cases -------------------------------------------------------------
small = linear_sieve(1)
check("sieve of 1 has no primes", small["primes"] == [])
check("sieve of 0 works", linear_sieve(0)["primes"] == [])

# --- the linear sieve strikes each composite exactly once ------------------
# (validated indirectly: every composite has a valid spf and the counts are right)
composites = [n for n in range(2, N + 1) if S["spf"][n] != n]
check("composite count + prime count = N-1",
      len(composites) + len(S["primes"]) == N - 1)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all linear_sieve tests passed")
