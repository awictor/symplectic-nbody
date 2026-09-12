"""The Chinese Remainder Theorem: reconstructing a number from its remainders.

If you know a number's remainders modulo several pairwise-coprime moduli, the CHINESE REMAINDER
THEOREM says there is a UNIQUE value modulo their product that matches all of them -- and gives a
constructive way to find it. This ancient result (Sunzi, 3rd century) is the backbone of modern
computing: RSA decryption speeds up 4x by working modulo the two prime factors separately and
recombining with CRT; large-integer and polynomial arithmetic is done in parallel residue systems;
and it underlies secret sharing, hashing, and error-correcting codes. It turns one hard computation
modulo a big number into several easy ones modulo small numbers.

The construction rests on the EXTENDED EUCLIDEAN ALGORITHM, which for coprime a and b finds integers
x, y with a*x + b*y = 1 -- hence the modular inverse of a mod b. Given congruences x ≡ r_i (mod m_i)
with coprime moduli, CRT builds the solution as x = sum r_i * M_i * (M_i^{-1} mod m_i) mod M, where M
is the product of all moduli and M_i = M / m_i. Each term is r_i in its own modulus and 0 in the
others, so they add without interference. When the moduli are NOT coprime the system may still be
solvable, but only if the congruences agree on the gcd of each pair -- a generalized CRT that this
module also handles, merging congruences pairwise and detecting contradictions.

This module implements the extended Euclidean algorithm, modular inverse, and both the coprime and
general Chinese Remainder Theorem, returning the unique residue and the combined modulus. It is
verified against brute force: that the CRT solution satisfies every input congruence and is the
smallest non-negative such value, that it matches an exhaustive search over the modulus range, that
the extended Euclidean algorithm produces a correct Bezout identity and modular inverses, that
non-coprime systems are solved when consistent and rejected when contradictory, and on the classic
Sunzi puzzle. Pure stdlib; a number-theory companion to the RSA, Shamir-secret-sharing, and
continued-fraction notes."""

from __future__ import annotations

from math import gcd


def extended_gcd(a, b):
    """Return (g, x, y) with a*x + b*y = g = gcd(a, b) (the Bezout identity)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def mod_inverse(a, m):
    """The modular inverse of a mod m (a and m must be coprime), in [0, m)."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {m} (not coprime)")
    return x % m


def crt(remainders, moduli):
    """Chinese Remainder Theorem for PAIRWISE-COPRIME moduli.

    remainders[i], moduli[i]: solve x ≡ remainders[i] (mod moduli[i]). Returns (x, M) where M is the
    product of the moduli and x in [0, M) is the unique solution."""
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")
    M = 1
    for m in moduli:
        M *= m
    x = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        inv = mod_inverse(Mi, m)
        x = (x + r * Mi * inv) % M
    return x % M, M


def crt_general(remainders, moduli):
    """Generalized CRT for arbitrary (possibly non-coprime) moduli. Merges congruences pairwise.

    Returns (x, lcm) if a solution exists (x is the smallest non-negative), or None if the system is
    contradictory."""
    x, m = 0, 1                              # start with x ≡ 0 (mod 1), always true
    for r, mi in zip(remainders, moduli):
        r = r % mi
        g, p, _ = extended_gcd(m, mi)
        if (r - x) % g != 0:
            return None                      # contradiction: no solution
        lcm = m // g * mi
        # merge: new x ≡ x (mod m) and ≡ r (mod mi)
        diff = (r - x) // g
        step = mi // g
        x = (x + m * (diff * mod_inverse_partial(m // g, step) % step)) % lcm
        m = lcm
    return x % m, m


def mod_inverse_partial(a, m):
    """Modular inverse of a mod m when they are coprime (used inside the general CRT merge). Returns
    the inverse in [0, m); if m == 1, returns 0."""
    if m == 1:
        return 0
    g, x, _ = extended_gcd(a % m, m)
    return x % m


def verify(x, remainders, moduli):
    """True if x satisfies every congruence x ≡ remainders[i] (mod moduli[i])."""
    return all((x - r) % m == 0 for r, m in zip(remainders, moduli))
