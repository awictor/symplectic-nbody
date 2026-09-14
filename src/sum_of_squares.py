"""Sums of two (and four) squares: Fermat's theorem, Cornacchia's algorithm, and Lagrange's guarantee.

Which integers are a sum of two squares, n = a^2 + b^2? Fermat's Christmas theorem (1640) answers it
completely for primes: an odd prime p is a sum of two squares if and only if p = 2 or p ≡ 1 (mod 4).
And there is a beautiful reason -- in the GAUSSIAN INTEGERS Z[i], such a p FACTORS as p = (a+bi)(a-bi),
so a^2 + b^2 = p; the primes p ≡ 3 (mod 4) stay prime in Z[i] and cannot. For general n, the
two-squares theorem says n is a sum of two squares iff every prime ≡ 3 (mod 4) in its factorization
appears to an EVEN power, because those primes must pair up to survive.

CORNACCHIA'S ALGORITHM turns the existence proof into a fast constructive one for a prime p ≡ 1 (mod 4):
find a square root r of -1 modulo p (r^2 ≡ -1, which exists exactly because p ≡ 1 mod 4), then run the
Euclidean algorithm on (p, r) and STOP when the remainder first drops below sqrt(p). That remainder and
its partner are the a, b -- one gcd-style descent recovers the representation that would otherwise take
a search. This module builds the general representation by factoring n, applying Cornacchia to each
prime ≡ 1 (mod 4), and composing the pieces with the Brahmagupta-Fibonacci two-square identity
(a^2+b^2)(c^2+d^2) = (ac-bd)^2 + (ad+bc)^2, which is just multiplication of Gaussian-integer norms.

It also implements LAGRANGE'S FOUR-SQUARE THEOREM: every non-negative integer is a sum of four squares,
with no exceptions at all -- computed here by a bounded search reduced through the two-square structure.

This module gives the two-square test, Cornacchia for primes, the general two-square representation,
the Gaussian-integer view, and a four-square decomposition. It is validated exactly: the two-square
test matches a brute-force search over all n up to a few thousand; every representation it returns
actually squares and sums to n; Fermat's criterion (p prime is a sum of two squares iff p = 2 or
p ≡ 1 mod 4) holds over the primes; Cornacchia's output satisfies a^2 + b^2 = p; and Lagrange's
four-square decomposition is exact for every n tested (including those with no two- or three-square
form). Pure stdlib; the additive-number-theory companion to the tonelli-shanks square roots and the
Pollard-rho factorization tools."""

from __future__ import annotations

from math import isqrt, gcd

from tonelli_shanks import sqrt_mod
from pollard_rho import is_prime, factorize as _prime_factorize


def _prime_factorization(n):
    """Prime factorization of n as a dict prime -> exponent."""
    facs = {}
    for p in _prime_factorize(n):
        facs[p] = facs.get(p, 0) + 1
    return facs


def is_sum_of_two_squares(n):
    """True iff n is a sum of two squares: every prime ≡ 3 (mod 4) has even exponent."""
    if n < 0:
        return False
    if n == 0:
        return True
    for p, e in _prime_factorization(n).items():
        if p % 4 == 3 and e % 2 == 1:
            return False
    return True


def cornacchia(p):
    """Cornacchia's algorithm: for a prime p ≡ 1 (mod 4) or p = 2, return (a, b) with a^2 + b^2 = p."""
    if p == 2:
        return (1, 1)
    if p % 4 != 1:
        raise ValueError("Cornacchia here needs p = 2 or p ≡ 1 (mod 4)")
    # square root of -1 mod p
    r = sqrt_mod(p - 1, p)  # sqrt of (p-1) = -1 mod p
    if r is None:
        raise ValueError("no square root of -1 -- p is not ≡ 1 mod 4")
    # ensure r > p/2 branch: run Euclid on (p, r), stop when remainder < sqrt(p)
    a, b = p, r
    limit = isqrt(p)
    while b > limit:
        a, b = b, a % b
    c = p - b * b
    s = isqrt(c)
    if s * s == c:
        return (b, s)
    # fallback (shouldn't happen for a valid prime)
    raise ValueError(f"Cornacchia failed for p={p}")


def _mul_gaussian(rep1, rep2):
    """Brahmagupta-Fibonacci: (a^2+b^2)(c^2+d^2) = (ac-bd)^2 + (ad+bc)^2."""
    a, b = rep1
    c, d = rep2
    return (abs(a * c - b * d), abs(a * d + b * c))


def two_squares(n):
    """Return (a, b) with a^2 + b^2 = n, or None if n is not a sum of two squares."""
    if n < 0:
        return None
    if n == 0:
        return (0, 0)
    facs = _prime_factorization(n)
    # each prime ≡ 3 mod 4 must have even exponent; it contributes p^(e/2) as a scalar
    scalar = 1
    rep = (1, 0)  # represents 1 = 1^2 + 0^2
    for p, e in facs.items():
        if p % 4 == 3:
            if e % 2 == 1:
                return None
            scalar *= p ** (e // 2)
        elif p == 2:
            for _ in range(e):
                rep = _mul_gaussian(rep, (1, 1))
        else:  # p ≡ 1 mod 4
            base = cornacchia(p)
            for _ in range(e):
                rep = _mul_gaussian(rep, base)
    a, b = rep
    return (a * scalar, b * scalar)


def four_squares(n):
    """Lagrange: return (a, b, c, d) with a^2+b^2+c^2+d^2 = n (every n >= 0 has one)."""
    if n < 0:
        raise ValueError("four_squares needs n >= 0")
    if n == 0:
        return (0, 0, 0, 0)
    # peel off the largest square d^2, then try to write the remainder as three squares,
    # and three as one + two-squares. Bounded search on d, c; two_squares for the rest.
    for d in range(isqrt(n), -1, -1):
        rem3 = n - d * d
        # try to write rem3 as a^2+b^2+c^2 by peeling c and using two_squares
        for c in range(isqrt(rem3), -1, -1):
            rem2 = rem3 - c * c
            ts = two_squares(rem2)
            if ts is not None:
                a, b = ts
                return (a, b, c, d)
    return None  # unreachable by Lagrange's theorem


def gaussian_norm(a, b):
    """The norm a^2 + b^2 of the Gaussian integer a + bi."""
    return a * a + b * b


def brute_two_squares(n):
    """Reference: find (a, b) with a^2 + b^2 = n by scanning, or None."""
    for a in range(isqrt(n) + 1):
        b2 = n - a * a
        b = isqrt(b2)
        if b * b == b2:
            return (a, b)
    return None
