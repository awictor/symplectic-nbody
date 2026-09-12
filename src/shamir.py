"""Shamir's secret sharing: splitting a secret so any k of n pieces reconstruct it.

How do you store a secret -- a master key, a launch code -- so that no single person holds it, yet any
sufficiently large group can recover it? SHAMIR'S SECRET SHARING (Adi Shamir, 1979) solves this with
a beautiful idea from geometry: a polynomial of degree k-1 is uniquely determined by any k of its
points, but k-1 points reveal NOTHING about it. Hide the secret as the constant term of a random
degree-(k-1) polynomial over a finite field, hand each participant one point (x, f(x)) as their SHARE,
and then any k shares interpolate the polynomial back and read off the secret, while any k-1 shares
leave every possible secret equally likely -- information-theoretic security, not merely
computational.

The arithmetic lives in a FINITE FIELD GF(p) (integers mod a prime) so that division is exact and the
scheme is perfectly secure with no rounding. To SHARE a secret s with threshold k: pick random
coefficients a_1..a_{k-1}, form f(x) = s + a_1 x + ... + a_{k-1} x^{k-1} mod p, and give participant i
the point (i, f(i)). To RECONSTRUCT from any k shares: LAGRANGE INTERPOLATION evaluates the unique
degree-(k-1) polynomial through those points at x=0, which is exactly f(0) = s -- computed with
modular inverses so it is exact. The scheme is (k, n)-THRESHOLD: any k of the n shares suffice, any
fewer are useless, and shares can be added or refreshed without changing the secret.

This module implements (k, n) secret sharing over a large prime field: splitting a secret integer
into shares and reconstructing it by Lagrange interpolation, with a helper for byte-string secrets.
It is verified that any k of the n shares reconstruct the secret exactly, that DIFFERENT subsets of k
shares all give the same secret, that k-1 shares reconstruct the wrong value (revealing nothing),
that the shares themselves look random, that a byte-string secret round-trips, and on the boundary
cases k=1 and k=n. Pure stdlib; a cryptography companion to the RSA, Diffie-Hellman, and
finite-field notes."""

from __future__ import annotations

# a 257-bit prime (2^256 + 297), large enough for 256-bit secrets
_PRIME = 2 ** 256 + 297


class _RNG:
    def __init__(self, seed=1):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def randbelow(self, n):
        # build a random integer < n from 64-bit LCG chunks
        bits = n.bit_length()
        out = 0
        while bits > 0:
            self.state = (6364136223846793005 * self.state + 1442695040888963407) & ((1 << 64) - 1)
            out = (out << 64) | self.state
            bits -= 64
        return out % n


def _eval_poly(coeffs, x, prime):
    """Evaluate a polynomial (coeffs[0] + coeffs[1] x + ...) at x, mod prime (Horner)."""
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % prime
    return acc


def split(secret, k, n, prime=_PRIME, seed=1):
    """Split an integer `secret` into `n` shares such that any `k` reconstruct it.

    Returns a list of (x, y) shares with x = 1..n. Requires 0 <= secret < prime and 1 <= k <= n."""
    if not (1 <= k <= n):
        raise ValueError("need 1 <= k <= n")
    if not (0 <= secret < prime):
        raise ValueError("secret must be in [0, prime)")
    rng = _RNG(seed)
    # random polynomial with the secret as the constant term
    coeffs = [secret] + [rng.randbelow(prime) for _ in range(k - 1)]
    return [(x, _eval_poly(coeffs, x, prime)) for x in range(1, n + 1)]


def _mod_inverse(a, p):
    """Modular inverse of a mod p (p prime) via the extended Euclidean algorithm."""
    return pow(a % p, p - 2, p)


def reconstruct(shares, prime=_PRIME):
    """Reconstruct the secret from a list of (x, y) shares by Lagrange interpolation at x=0.

    Any `k` shares of a (k, n) split return the secret; fewer return a wrong value."""
    secret = 0
    xs = [s[0] for s in shares]
    for i, (xi, yi) in enumerate(shares):
        # Lagrange basis L_i(0) = prod_{j != i} (-x_j) / (x_i - x_j)
        num = 1
        den = 1
        for j, xj in enumerate(xs):
            if i == j:
                continue
            num = (num * (-xj)) % prime
            den = (den * (xi - xj)) % prime
        term = (yi * num % prime) * _mod_inverse(den, prime) % prime
        secret = (secret + term) % prime
    return secret % prime


def split_bytes(secret_bytes, k, n, prime=_PRIME, seed=1):
    """Split a byte-string secret (encoded as a big integer) into shares."""
    s = int.from_bytes(secret_bytes, "big")
    if s >= prime:
        raise ValueError("secret too large for the field; use a bigger prime")
    length = len(secret_bytes)
    return length, split(s, k, n, prime=prime, seed=seed)


def reconstruct_bytes(length, shares, prime=_PRIME):
    """Reconstruct a byte-string secret of known length from shares."""
    s = reconstruct(shares, prime=prime)
    return s.to_bytes(length, "big")
