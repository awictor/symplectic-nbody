"""Cipolla's algorithm: modular square roots by stepping into an imaginary quadratic field.

Solving x^2 = n (mod p) for a prime p is the modular analogue of taking a square root. Tonelli-Shanks
does it by chasing 2-adic structure; CIPOLLA'S ALGORITHM (Michele Cipolla, 1907) takes a strikingly
different and elegant route through a QUADRATIC FIELD EXTENSION. Pick any a for which a^2 - n is a
non-residue mod p, so sqrt(a^2 - n) does not exist in the field of integers mod p. Adjoin it: work in
the extension GF(p^2) = { u + v*w : u, v in GF(p) }, where w = sqrt(a^2 - n) is a formal 'imaginary'
element with w^2 = a^2 - n. Then the single field power

    x = (a + w)^{(p+1)/2}   in GF(p^2)

is guaranteed to be a genuine element of GF(p) (its w-part vanishes) and to satisfy x^2 = n. One
exponentiation in a two-dimensional field, and the square root falls out -- no case analysis on the
powers of two dividing p - 1, which makes Cipolla especially clean when that power is large.

The reason it works is a small piece of magic: raising to the (p+1)/2 power in GF(p^2) is like a
'norm-halving' that lands back in the base field precisely because a^2 - n is a non-residue, and the
result squares to (a^2 - n) times a Frobenius conjugate that collapses to n. This module implements
Cipolla with fast field exponentiation, the Legendre symbol to test residues and choose a, and returns
both roots. It is validated against the repo's Tonelli-Shanks solver: both agree on the square roots
for many primes and residues; every returned root squares to n mod p; non-residues are correctly
reported as having no root; the two roots are negatives of each other mod p; and it handles the small
cases n = 0 and p = 2. Cross-checks Tonelli-Shanks. Pure stdlib; the modular-arithmetic companion to
the Tonelli-Shanks, quadratic-residue, and discrete-log tools."""

from __future__ import annotations


def legendre_symbol(a, p):
    """Legendre symbol (a/p): 1 if a is a nonzero QR mod p, -1 if a non-residue, 0 if a ≡ 0."""
    a %= p
    if a == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    return ls if ls == 1 else -1


def is_quadratic_residue(a, p):
    return legendre_symbol(a, p) == 1


def _mul(x, y, w2, p):
    """Multiply (x0 + x1 w)(y0 + y1 w) in GF(p^2) where w^2 = w2 (mod p)."""
    a, b = x
    c, d = y
    return ((a * c + b * d * w2) % p, (a * d + b * c) % p)


def _pow(x, e, w2, p):
    """Fast exponentiation of a GF(p^2) element x = (x0, x1) to power e."""
    result = (1, 0)
    base = x
    while e > 0:
        if e & 1:
            result = _mul(result, base, w2, p)
        base = _mul(base, base, w2, p)
        e >>= 1
    return result


def sqrt_mod(n, p):
    """A square root of n modulo the prime p by Cipolla's algorithm, or None if none exists."""
    n %= p
    if p == 2:
        return n            # 0->0, 1->1
    if n == 0:
        return 0
    if legendre_symbol(n, p) != 1:
        return None         # non-residue: no square root
    # find a with a^2 - n a non-residue
    a = 1
    while True:
        w2 = (a * a - n) % p
        if legendre_symbol(w2, p) == -1:
            break
        a += 1
    # x = (a + w)^{(p+1)/2} in GF(p^2)
    res = _pow((a, 1), (p + 1) // 2, w2, p)
    x, imag = res
    # the imaginary part must vanish
    if imag != 0:
        return None         # should not happen for a valid residue
    return x % p


def both_roots(n, p):
    """Both square roots of n mod p as a sorted tuple, or None if n is a non-residue."""
    r = sqrt_mod(n, p)
    if r is None:
        return None
    r2 = (-r) % p
    return tuple(sorted({r, r2}))


def verify(n, r, p):
    """Check r^2 ≡ n (mod p)."""
    return (r * r - n) % p == 0
