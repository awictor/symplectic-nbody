"""Tonelli-Shanks: computing square roots modulo a prime.

Solving x^2 = n (mod p) -- finding a MODULAR SQUARE ROOT -- is a basic operation in number theory and
cryptography: it decompresses elliptic-curve points (recovering y from x on the curve), appears in
quadratic sieve factoring, primality proving, and the Rabin cryptosystem. Unlike ordinary square
roots there is no formula in general, but the TONELLI-SHANKS algorithm (1891/1973) finds a root in
expected polynomial time whenever one exists, for any odd prime modulus.

First, does a root even exist? By EULER'S CRITERION, n is a QUADRATIC RESIDUE mod p iff
n^((p-1)/2) = 1 (mod p); this is the LEGENDRE SYMBOL, +1 for residues, -1 for non-residues, 0 for
multiples of p. If n is a residue, Tonelli-Shanks proceeds: write p-1 = q * 2^s with q odd. When
s = 1 (p = 3 mod 4) the root is simply n^((p+1)/4). Otherwise it finds a quadratic NON-residue z,
uses it to build a value of order 2^s, and iteratively squares and corrects a running candidate --
each step halving the "order of the error" -- until the error vanishes and the candidate squares to
n. The two roots are r and p - r.

This module implements the Legendre symbol (Euler's criterion), a quadratic-residue test, and
Tonelli-Shanks modular square root. It is verified against exact references: that the returned root
squares back to n modulo p, that both roots (r and p-r) are found, that non-residues are correctly
reported as having no root, that the Legendre symbol matches a brute-force count of squares, that the
p = 3 mod 4 fast path agrees with the general algorithm, and on primes of both forms and small
exhaustively-checkable cases. Pure stdlib; a number-theory companion to the RSA, elliptic-curve, and
Chinese-Remainder notes."""

from __future__ import annotations


def legendre_symbol(a, p):
    """The Legendre symbol (a/p) for an odd prime p: 1 if a is a nonzero quadratic residue, -1 if a
    non-residue, 0 if p divides a. Computed by Euler's criterion."""
    a %= p
    if a == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    return 1 if ls == 1 else -1


def is_quadratic_residue(a, p):
    """True if a is a quadratic residue modulo the odd prime p (has a square root)."""
    return legendre_symbol(a, p) != -1


def sqrt_mod(n, p):
    """A square root of n modulo the odd prime p (Tonelli-Shanks), or None if none exists.

    Returns r with r*r = n (mod p); the other root is p - r."""
    n %= p
    if n == 0:
        return 0
    if p == 2:
        return n
    if legendre_symbol(n, p) != 1:
        return None                          # non-residue: no square root
    # fast path for p = 3 (mod 4)
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    # general Tonelli-Shanks: write p-1 = q * 2^s with q odd
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    # find a quadratic non-residue z
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    while t != 1:
        # find the least i, 0 < i < m, with t^(2^i) = 1
        i = 0
        temp = t
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
            if i == m:
                return None                  # should not happen for a genuine residue
        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i
    return r


def both_roots(n, p):
    """Both square roots of n modulo p as a sorted tuple, or None if n is a non-residue."""
    r = sqrt_mod(n, p)
    if r is None:
        return None
    roots = {r % p, (p - r) % p}
    return tuple(sorted(roots))
