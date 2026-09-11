"""Diffie-Hellman: agreeing on a secret in the open.

In 1976 Whitfield Diffie and Martin Hellman solved a problem that had seemed impossible: two
people who have never met, talking over a wiretapped line, can agree on a shared secret that the
eavesdropper cannot learn. It was the birth of public-key cryptography.

The trick rests on the discrete logarithm. Fix a large prime p and a generator g. Alice picks a
secret a and sends g^a mod p; Bob picks a secret b and sends g^b mod p. Each raises what they
received to their own secret:

    Alice computes (g^b)^a = g^{ab} mod p,
    Bob   computes (g^a)^b = g^{ab} mod p,

so they arrive at the SAME number g^{ab} mod p -- the shared secret -- while the eavesdropper
sees only g^a and g^b. To steal the secret they would have to recover a from g^a mod p, the
discrete-logarithm problem, which is easy to state and (for well-chosen p) astronomically hard
to solve. Modular exponentiation is a one-way function: fast forward, infeasible to invert.

The catch is authentication: without it, an active attacker can sit in the middle, doing a
separate exchange with each side (the man-in-the-middle attack) -- which is why real protocols
sign the exchanged values. This module generates safe-prime parameters, finds a generator,
performs the exchange, confirms both parties derive the same secret, and includes a
baby-step/giant-step discrete-log solver to show the exchange is correct while brute force scales
as sqrt(p). Pure stdlib; the key-agreement companion to the RSA note.
"""

from __future__ import annotations

import math


def modexp(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation base^exp mod mod by square-and-multiply."""
    result = 1
    base %= mod
    while exp:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result


_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def is_probable_prime(n: int) -> bool:
    """Miller-Rabin primality test with the standard small-prime witnesses."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _SMALL_PRIMES:
        x = modexp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _factorize(n: int):
    """Trial-division prime factors of n (small n only -- used on p-1 to test generators)."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def is_generator(g: int, p: int) -> bool:
    """True if g is a primitive root mod p: its powers run through every nonzero residue. Checked
    via g^((p-1)/q) != 1 for each prime factor q of p-1."""
    if not 1 < g < p:
        return False
    order = p - 1
    for q in _factorize(order):
        if modexp(g, order // q, p) == 1:
            return False
    return True


def find_generator(p: int) -> int:
    """Find the smallest primitive root modulo the prime p."""
    for g in range(2, p):
        if is_generator(g, p):
            return g
    raise ValueError("no generator found (is p prime?)")


class _Rng:
    """Seeded LCG; high bits. Reproducible parameter/secret choice for demos -- NOT a
    cryptographically secure source; a real system uses os.urandom."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def randrange(self, lo: int, hi: int) -> int:
        """A pseudorandom integer in [lo, hi)."""
        span = hi - lo
        # gather enough bits
        val = 0
        for _ in range(4):
            val = (val << 16) | (self._next() >> 16)
        return lo + (val % span)


def generate_safe_prime(nbits: int, rng: _Rng) -> int:
    """Generate a safe prime p = 2q + 1 (with q also prime) of about nbits bits. Safe primes make
    every non-trivial element a generator or of large order, foiling small-subgroup attacks."""
    while True:
        q = rng.randrange(1 << (nbits - 2), 1 << (nbits - 1)) | 1
        if is_probable_prime(q):
            p = 2 * q + 1
            if is_probable_prime(p):
                return p


def make_parameters(nbits: int = 16, seed: int = 1):
    """Public Diffie-Hellman parameters (p, g): a safe prime and a generator."""
    rng = _Rng(seed)
    p = generate_safe_prime(nbits, rng)
    g = find_generator(p)
    return p, g


def public_key(secret: int, p: int, g: int) -> int:
    """The value a party broadcasts: g^secret mod p."""
    return modexp(g, secret, p)


def shared_secret(their_public: int, my_secret: int, p: int) -> int:
    """The agreed secret from the other party's public value and my secret: their_public^mine."""
    return modexp(their_public, my_secret, p)


def exchange(a_secret: int, b_secret: int, p: int, g: int):
    """Run a full exchange. Returns (A, B, secret_alice, secret_bob); the two secrets must match."""
    A = public_key(a_secret, p, g)
    B = public_key(b_secret, p, g)
    sa = shared_secret(B, a_secret, p)
    sb = shared_secret(A, b_secret, p)
    return A, B, sa, sb


def discrete_log_bsgs(g: int, h: int, p: int):
    """Solve g^x = h (mod p) for x by baby-step/giant-step in O(sqrt(p)) time and space. Returns
    x, or None if no solution. This is the attack an eavesdropper would run -- feasible only
    because p is small here; for cryptographic p it is astronomically expensive."""
    n = 1 + int(math.isqrt(p - 1))
    # baby steps: g^j for j in [0, n)
    table = {}
    e = 1
    for j in range(n):
        table.setdefault(e, j)
        e = (e * g) % p
    # giant steps: h * (g^-n)^i
    factor = modexp(modexp(g, p - 2, p), n, p)  # g^{-n} mod p (Fermat inverse, p prime)
    gamma = h % p
    for i in range(n):
        if gamma in table:
            return i * n + table[gamma]
        gamma = (gamma * factor) % p
    return None
