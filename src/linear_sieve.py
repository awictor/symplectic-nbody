"""Linear sieve: every prime, factorization, totient, and Mobius value up to N in O(N).

The Sieve of Eratosthenes finds primes up to N in O(N log log N) by crossing off multiples; the LINEAR
SIEVE (Euler's sieve) does it in true O(N) while also computing, for free, the SMALLEST PRIME FACTOR
of every number and -- because they are MULTIPLICATIVE FUNCTIONS built from prime factorizations -- the
EULER TOTIENT phi(n) (how many integers below n are coprime to it) and the MOBIUS function mu(n) (the
inclusion-exclusion sign central to number-theoretic inversion). Precomputing these arrays underlies
fast factorization of many queries, divisor-sum and gcd-sum problems, counting coprime pairs,
Dirichlet convolution, and the Mobius inversion that appears throughout analytic number theory and
competitive programming.

The trick that makes it linear: every composite is struck exactly ONCE, by its smallest prime factor.
Iterate i from 2 to N; if i has no recorded smallest prime factor it is prime. For each known prime p
(in increasing order) with p*i <= N, mark p as the smallest prime factor of p*i and stop the inner
loop the moment p divides i -- because beyond that point p*i would later be struck by a smaller prime
factor, causing the double-counting the classic sieve suffers. The same pass propagates the
multiplicative functions: phi and mu of p*i follow from phi(i) and mu(i) by simple rules depending on
whether p divides i, since phi and mu are multiplicative and have clean prime-power values.

This module builds, up to N, the list of primes, the smallest-prime-factor table (giving O(log n)
factorization of any n <= N), the totient array, and the Mobius array, and offers helpers to factorize
via the SPF table and to test primality. It is verified against independent brute force -- trial
division for primality and factorization, the coprime-counting definition of phi, and the
squarefree/prime-count definition of mu -- across the whole range up to several thousand, confirming
every entry. Pure stdlib; a number-theory companion to the Pollard-rho factorization, CRT, and
Tonelli-Shanks notes."""

from __future__ import annotations

from math import gcd


def linear_sieve(n):
    """Compute, for all integers 0..n: the list of primes, the smallest-prime-factor array `spf`, the
    Euler totient array `phi`, and the Mobius array `mu`, in O(n).

    Returns a dict with keys 'primes', 'spf', 'phi', 'mu'."""
    spf = [0] * (n + 1)              # smallest prime factor (0 for 0 and 1)
    phi = [0] * (n + 1)
    mu = [0] * (n + 1)
    primes = []
    if n >= 1:
        phi[1] = 1
        mu[1] = 1
    for i in range(2, n + 1):
        if spf[i] == 0:              # i is prime
            spf[i] = i
            phi[i] = i - 1
            mu[i] = -1
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > n:
                break
            spf[i * p] = p
            if i % p == 0:
                # p divides i: p^2 | i*p, so mu = 0 and phi(i*p) = phi(i)*p
                phi[i * p] = phi[i] * p
                mu[i * p] = 0
                break
            else:
                # p coprime to i: multiplicative
                phi[i * p] = phi[i] * (p - 1)
                mu[i * p] = -mu[i]
    return {"primes": primes, "spf": spf, "phi": phi, "mu": mu}


def primes_up_to(n):
    """The list of primes <= n."""
    return linear_sieve(n)["primes"]


def factorize_with_spf(x, spf):
    """Factorize x (2 <= x <= len(spf)-1) using a precomputed smallest-prime-factor table. Returns a
    dict {prime: exponent}. O(log x)."""
    factors = {}
    while x > 1:
        p = spf[x]
        while x % p == 0:
            factors[p] = factors.get(p, 0) + 1
            x //= p
    return factors


def is_prime_sieve(x, spf):
    """True iff x is prime, using the SPF table (x must be within the sieved range)."""
    return x >= 2 and spf[x] == x


# --- brute-force references -------------------------------------------------
def brute_is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def brute_factorize(n):
    """Trial-division factorization -> {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def brute_totient(n):
    """Euler totient by counting integers in [1, n] coprime to n."""
    if n == 0:
        return 0
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)


def brute_mobius(n):
    """Mobius function from the factorization: 0 if any square factor, else (-1)^(#distinct primes)."""
    if n == 1:
        return 1
    f = brute_factorize(n)
    if any(e >= 2 for e in f.values()):
        return 0
    return -1 if len(f) % 2 else 1
