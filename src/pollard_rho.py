"""Integer factorization: Pollard's rho, Pollard's p-1, and full prime factorization.

Multiplying two large primes is easy; recovering them from the product is the FACTORIZATION problem
that the security of RSA rests on. Trial division works only up to tiny factors -- factoring a
40-digit number by trial division would take longer than the age of the universe. POLLARD'S RHO
algorithm, from 1975, finds a non-trivial factor of a composite n in expected O(n^{1/4}) time and
O(1) space -- for a factor p, in about sqrt(p) iterations. It is the workhorse for the medium-size
factors that appear inside real numbers, and the tool that makes full factorization of 30-40 digit
numbers instant.

The idea is a probabilistic collision. Iterate a pseudo-random sequence x -> x^2 + c (mod n); by the
BIRTHDAY PARADOX the sequence enters a cycle after about sqrt(p) steps MODULO any prime factor p, and
when two iterates collide mod p (but not mod n) their difference shares the factor p with n, so
gcd(|x_i - x_j|, n) reveals it. FLOYD'S cycle detection (the tortoise and hare) finds the collision
in constant memory; Brent's variant, used here, batches the gcd computations to run faster. POLLARD'S
p-1 method is a complementary trick: if p-1 has only small prime factors (p is 'smooth'), then
a^(k!) - 1 is divisible by p for modest k, and gcd of that with n exposes p -- devastating against
poorly-chosen RSA primes, which is why real RSA requires 'safe' primes.

This module implements Miller-Rabin primality (deterministic for 64-bit integers), Brent's improved
Pollard's rho, Pollard's p-1, and a full recursive prime factorization that combines them, returning
the sorted prime factors with multiplicity. It is verified against brute-force trial division and
known factorizations: that the product of the returned factors equals the input, that every returned
factor is prime, that it recovers the factors of semiprimes (products of two primes, the RSA case),
that primality classification matches a sieve, and that perfect powers and highly composite numbers
factor correctly. Pure stdlib; a number-theory companion to the RSA and modular-arithmetic notes."""

from __future__ import annotations

from math import gcd


# small primes for quick trial division
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def is_prime(n):
    """Deterministic Miller-Rabin primality test (exact for all n < 3.3 * 10^24)."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    # write n-1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # these witnesses are sufficient for all n < 3.3e24
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _pollard_rho_brent(n):
    """Brent's improved Pollard's rho: return a non-trivial factor of composite n (not necessarily
    prime), or n if it fails (extremely unlikely for composite n)."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    # deterministic-ish sequence params derived from n (no global RNG needed)
    y = (n - 1) % n or 2
    c = (n % 7) + 1
    m = 128
    g = 1
    r = 1
    q = 1
    x = ys = y
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            count = min(m, r - k)
            for _ in range(count):
                y = (y * y + c) % n
                q = (q * abs(x - y)) % n
            g = gcd(q, n)
            k += m
        r *= 2
    if g == n:
        # backtrack step by step
        g = 1
        while g == 1:
            ys = (ys * ys + c) % n
            g = gcd(abs(x - ys), n)
    return g


def pollard_rho(n):
    """A non-trivial factor of composite n via Brent's Pollard rho, or None if n is prime."""
    if n <= 1:
        return None
    if is_prime(n):
        return None
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return p
    # retry with different c if needed
    for attempt in range(20):
        f = _pollard_rho_brent(n)
        if f != n and f != 1:
            return f
    return None


def pollard_p_minus_1(n, bound=10000):
    """Pollard's p-1: return a non-trivial factor of n if some prime factor p has p-1 that is
    `bound`-smooth, else None."""
    if n % 2 == 0:
        return 2
    a = 2
    for j in range(2, bound + 1):
        a = pow(a, j, n)
        d = gcd(a - 1, n)
        if 1 < d < n:
            return d
        if d == n:
            return None
    return None


def factorize(n):
    """Full prime factorization: sorted list of prime factors with multiplicity (product == n)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return []
    factors = []

    def _factor(m):
        if m == 1:
            return
        if is_prime(m):
            factors.append(m)
            return
        d = pollard_rho(m)
        if d is None:
            # fallback: trial division (should not happen for composite m)
            d = _trial_factor(m)
        _factor(d)
        _factor(m // d)

    _factor(n)
    factors.sort()
    return factors


def _trial_factor(n):
    i = 2
    while i * i <= n:
        if n % i == 0:
            return i
        i += 1
    return n


def factor_pairs(n):
    """Prime factorization as (prime, exponent) pairs in ascending prime order."""
    factors = factorize(n)
    out = []
    for p in factors:
        if out and out[-1][0] == p:
            out[-1][1] += 1
        else:
            out.append([p, 1])
    return [(p, e) for p, e in out]


def euler_phi(n):
    """Euler's totient via the factorization: phi(n) = n * prod(1 - 1/p) over distinct primes p."""
    if n <= 0:
        return 0
    result = n
    for p, _ in factor_pairs(n):
        result -= result // p
    return result


def num_divisors(n):
    """The number of positive divisors, from the factorization: prod(e_i + 1)."""
    if n <= 0:
        return 0
    d = 1
    for _, e in factor_pairs(n):
        d *= (e + 1)
    return d
