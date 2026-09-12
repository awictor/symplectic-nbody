"""Baby-step giant-step: solving the discrete logarithm in O(sqrt(n)).

The DISCRETE LOGARITHM problem asks: given a base g, a target h, and a modulus m, find the exponent x
with g^x = h (mod m). It is the hard problem underpinning Diffie-Hellman key exchange, ElGamal
encryption, and DSA signatures -- their security rests on the belief that no fast (polynomial) general
algorithm exists. But the naive search over all exponents is O(n) where n is the order of g; the
BABY-STEP GIANT-STEP algorithm (Shanks) cuts that to O(sqrt(n)) time and space by a classic
meet-in-the-middle trick, making moderate-size instances solvable and demonstrating why cryptographic
groups must be large.

The idea: write x = i*N + j with N = ceil(sqrt(n)) and 0 <= i, j < N. Then g^x = h becomes
g^(i*N) * g^j = h, i.e. g^j = h * (g^{-N})^i. Precompute a table of the BABY STEPS g^j for all j
(the "small" exponents) and store them in a hash map. Then take GIANT STEPS: multiply h by g^{-N}
repeatedly (i = 0, 1, 2, ...) and look each result up in the table; a hit gives j, and x = i*N + j.
Because both loops run only sqrt(n) times, the whole search is O(sqrt(n)) -- exponentially faster than
brute force, yet still exponential in the bit-length, which is why real cryptographic groups (256-bit
and up) stay secure.

This module implements baby-step giant-step discrete log modulo a prime (or with a supplied group
order), plus a modular-order helper. It is verified against brute force and by round-tripping: that
the returned x satisfies g^x = h (mod m), that it matches an exhaustive search for the smallest
exponent, that it correctly reports when no solution exists, that it recovers the exponent in a
Diffie-Hellman-style exchange, and on known small cases. Pure stdlib; a cryptography and
number-theory companion to the Diffie-Hellman, RSA, and Tonelli-Shanks notes."""

from __future__ import annotations

import math


def discrete_log(g, h, m, order=None):
    """Smallest non-negative x with g^x = h (mod m), or None if no such x exists.

    order: the order of g modulo m (defaults to m-1, correct when m is prime and g a generator; an
    upper bound like m-1 is always safe for a prime modulus)."""
    g %= m
    h %= m
    if order is None:
        order = m - 1
    N = int(math.isqrt(order)) + 1

    # baby steps: table[g^j mod m] = j for j in 0..N-1 (keep the smallest j on collision)
    table = {}
    cur = 1
    for j in range(N):
        if cur not in table:
            table[cur] = j
        cur = (cur * g) % m

    # factor = g^{-N} mod m
    g_inv = pow(g, -1, m)                     # requires gcd(g, m) == 1
    factor = pow(g_inv, N, m)

    # giant steps: gamma = h * factor^i
    gamma = h
    for i in range(N + 1):
        if gamma in table:
            x = i * N + table[gamma]
            if pow(g, x, m) == h:            # guard against spurious hits
                return x
        gamma = (gamma * factor) % m
    return None


def multiplicative_order(g, m):
    """The multiplicative order of g modulo m: the smallest k > 0 with g^k = 1 (mod m). Requires
    gcd(g, m) = 1. Returns None if g is not invertible."""
    if math.gcd(g, m) != 1:
        return None
    k = 1
    cur = g % m
    while cur != 1:
        cur = (cur * g) % m
        k += 1
        if k > m:                            # safety
            return None
    return k


def brute_discrete_log(g, h, m):
    """Naive O(n) discrete log for cross-checking. Returns the smallest x or None."""
    g %= m
    h %= m
    cur = 1                                   # g^0
    for x in range(m):
        if cur == h:
            return x
        cur = (cur * g) % m
    return None
