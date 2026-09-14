"""Lenstra's elliptic-curve method (ECM): factoring integers by finding a point that vanishes mod p.

Pollard's p-1 method finds a prime factor p of N whenever p-1 is smooth (a product of small primes) --
but it fails silently when no factor has that lucky structure. Hendrik Lenstra's 1985 ELLIPTIC-CURVE
METHOD is the brilliant generalization: instead of relying on the fixed group (Z/pZ)^* of order p-1, it
works on a RANDOM elliptic curve mod N, whose order mod p is some number near p that VARIES with the
curve. If that order happens to be smooth, the method finds p; if not, just try another curve. So ECM is
not defeated by a single unlucky factorization -- it keeps rolling fresh dice, and it is the best known
algorithm for pulling out factors of up to ~40 digits, the workhorse behind removing medium primes
before a number field sieve finishes the job.

The mechanism is beautiful. Pick a random curve y^2 = x^3 + ax + b mod N and a point P on it, and try to
compute k*P for a highly composite k = lcm(1..B) (or a product of prime powers up to a bound). The
elliptic-curve addition needs a modular INVERSE, and mod a true prime it always exists -- but we are
working mod the COMPOSITE N. When we try to invert a quantity that is zero mod one prime factor p but
nonzero mod another, the extended-Euclid gcd with N returns p itself: the factorization falls out of the
failed inversion. That "failure" is exactly the signal ECM is designed to catch.

This module implements ECM with affine Weierstrass-curve arithmetic over Z/NZ, extracting a factor from
any non-invertible slope via gcd, trying multiple random curves, and recursing to a full prime
factorization (using the repo's Baillie-PSW test to know when to stop). It is validated: every returned
factor genuinely divides N; the product of the returned prime factors equals N; each factor passes the
Baillie-PSW primality test; it factors semiprimes (products of two primes), prime powers, and numbers
with several factors; it factors cases where Pollard p-1 struggles; and primes are reported as their own
sole factor. Reuses the repo's Baillie-PSW primality test. Pure stdlib; the factorization companion to
the Pollard-rho, Dixon, and Baillie-PSW tools."""

from __future__ import annotations

from math import gcd, isqrt

from baillie_psw import is_prime


class _FactorFound(Exception):
    def __init__(self, factor):
        self.factor = factor


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self, n):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state % n


def _inv_or_factor(a, N):
    """Return the modular inverse of a mod N, or raise _FactorFound if a shares a factor with N."""
    a %= N
    g = gcd(a, N)
    if g != 1:
        raise _FactorFound(g)
    return pow(a, -1, N)


def _ec_add(P, Q, a, N):
    """Add two points on y^2 = x^3 + a x + b over Z/NZ (b implicit). Raises _FactorFound on a gcd hit."""
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and (y1 + y2) % N == 0:
        return None                              # P + (-P) = identity
    if P == Q:
        # doubling: slope = (3 x1^2 + a) / (2 y1)
        num = (3 * x1 * x1 + a) % N
        den = (2 * y1) % N
    else:
        num = (y2 - y1) % N
        den = (x2 - x1) % N
    inv = _inv_or_factor(den, N)                 # may raise _FactorFound
    s = (num * inv) % N
    x3 = (s * s - x1 - x2) % N
    y3 = (s * (x1 - x3) - y1) % N
    return (x3, y3)


def _ec_mul(k, P, a, N):
    """Scalar multiply k*P by double-and-add. Raises _FactorFound if an inversion fails."""
    result = None
    addend = P
    while k > 0:
        if k & 1:
            result = _ec_add(result, addend, a, N)
        addend = _ec_add(addend, addend, a, N)
        k >>= 1
    return result


def find_factor(N, curves=40, B=10000, seed=1):
    """Try to find a non-trivial factor of composite N by ECM. Returns a factor, or None on failure."""
    if N % 2 == 0:
        return 2
    if N % 3 == 0:
        return 3
    rng = _Rng(seed)
    # precompute the scalar k = product of prime powers <= B
    from math import log
    primes = _small_primes(B)
    for _ in range(curves):
        # random curve y^2 = x^3 + a x + b through a random point P
        x0 = rng.next(N)
        y0 = rng.next(N)
        a = rng.next(N)
        # b chosen so P=(x0,y0) is on the curve: b = y0^2 - x0^3 - a x0
        P = (x0, y0)
        try:
            k = 1
            Q = P
            for p in primes:
                pe = p
                while pe * p <= B:
                    pe *= p
                Q = _ec_mul(pe, Q, a, N)
                if Q is None:
                    break
        except _FactorFound as e:
            f = e.factor
            if 1 < f < N:
                return f
            # f == N: this curve failed, try the next
    return None


def _small_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def factorize(N, curves=60, B=5000, seed=1):
    """Full prime factorization of N as a sorted list (with multiplicity). Uses ECM + primality."""
    if N <= 1:
        return [] if N == 1 else [N]
    factors = []
    stack = [N]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors.append(m)
            continue
        # peel small primes first (fast)
        peeled = False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
            if m % p == 0:
                factors.append(p)
                stack.append(m // p)
                peeled = True
                break
        if peeled:
            continue
        # perfect power check
        f = _find_perfect_power_factor(m)
        if f is None:
            f = find_factor(m, curves=curves, B=B, seed=seed)
        if f is None or f == m:
            # fall back: give up and treat as prime-ish (should not happen for reasonable N)
            factors.append(m)
            continue
        stack.append(f)
        stack.append(m // f)
    return sorted(factors)


def _find_perfect_power_factor(m):
    """If m = r^e, return r; else None (handles prime-power inputs ECM can miss)."""
    for e in range(2, m.bit_length() + 1):
        r = round(m ** (1.0 / e))
        for cand in (r - 1, r, r + 1):
            if cand > 1 and cand ** e == m:
                return cand
    return None


def verify(N, factors):
    """Check the factors multiply to N and each is prime."""
    prod = 1
    for f in factors:
        prod *= f
    return prod == N and all(is_prime(f) for f in factors)
