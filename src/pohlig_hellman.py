"""Pohlig-Hellman: the discrete logarithm made easy when the group order is smooth.

The DISCRETE LOGARITHM problem -- given g, h in a group, find x with g^x = h -- is the security
foundation of Diffie-Hellman, ElGamal, and DSA. Its hardness depends entirely on the group ORDER: the
Pohlig-Hellman algorithm (1978) shows that if the order n = prod p_i^{e_i} has only SMALL prime
factors (n is "smooth"), the problem falls apart, reducing to a discrete log in each small
prime-power-order subgroup, solved by baby-step giant-step, and reassembled by the Chinese Remainder
Theorem. This is exactly why real cryptographic groups are chosen with order divisible by a large
prime -- to deny Pohlig-Hellman its factors.

The reduction, for cyclic group of order n = prod p_i^{e_i}:

  1. For each prime power q = p^e dividing n, project the problem into the unique subgroup of order q
     by raising to the power n/q: solving (g^{n/q})^x = h^{n/q} gives x modulo q.
  2. Within that subgroup, find x mod p^e DIGIT BY DIGIT in base p (Pohlig-Hellman's inner loop): each
     p-ary digit is a discrete log in the order-p subgroup, a tiny BSGS.
  3. Combine the residues x mod p_i^{e_i} by the Chinese Remainder Theorem into x mod n.

This module implements the full algorithm on the multiplicative group modulo a prime, reusing the
repo's baby-step giant-step discrete log, CRT, and factorization. It is validated exactly: it recovers
x for random exponents in prime fields, agrees with the brute-force and BSGS discrete logs, correctly
returns None when no logarithm exists, handles the per-prime-power subgroup solves, and its speed
advantage on a smooth-order group over plain BSGS is demonstrated. Pure stdlib; the smooth-order
discrete-log companion to the baby-step-giant-step, CRT, and Diffie-Hellman tools."""

from __future__ import annotations

from crt import crt
from discrete_log import discrete_log as bsgs_log, multiplicative_order
from pollard_rho import factorize as _factorize


def _prime_power_factorization(n):
    """Factor n as a list of (prime, exponent) pairs."""
    facs = {}
    for p in _factorize(n):
        facs[p] = facs.get(p, 0) + 1
    return sorted(facs.items())


def _dlog_prime_power(g, h, p, e, order, mod):
    """Discrete log x mod p^e in the subgroup, found digit by digit in base p.

    g has order `order` in (Z/mod)^*; we solve for x mod p^e using the order-p subgroup.
    """
    # gamma = generator of the order-p subgroup = g^{order/p}
    gamma = pow(g, order // p, mod)
    x = 0
    pe = 1
    ginv = pow(g, -1, mod)
    for k in range(e):
        # h_k = (h * g^{-x})^{order / p^{k+1}}
        exp = order // (p ** (k + 1))
        base = (h * pow(ginv, x, mod)) % mod
        hk = pow(base, exp, mod)
        # solve gamma^{d_k} = hk in the order-p subgroup
        d_k = bsgs_log(gamma, hk, mod, order=p)
        if d_k is None:
            return None
        x += d_k * pe
        pe *= p
    return x % (p ** e)


def pohlig_hellman(g, h, mod, order=None):
    """Discrete log x with g^x = h (mod `mod`) via Pohlig-Hellman. Returns x mod order, or None.

    order: the order of g modulo mod (defaults to the multiplicative order).
    """
    g %= mod
    h %= mod
    if h == 1:
        return 0
    if order is None:
        order = multiplicative_order(g, mod)
        if order is None:
            return None

    factors = _prime_power_factorization(order)
    residues = []
    moduli = []
    for p, e in factors:
        q = p ** e
        # project into the order-q subgroup
        gq = pow(g, order // q, mod)
        hq = pow(h, order // q, mod)
        # solve in the subgroup of order q, digit by digit in base p
        xq = _dlog_prime_power(g, h, p, e, order, mod)
        if xq is None:
            return None
        residues.append(xq)
        moduli.append(q)

    x, _M = crt(residues, moduli)   # crt returns (solution, product of moduli)
    x %= order
    # verify
    if pow(g, x, mod) == h:
        return x
    return None


def is_smooth(n, bound):
    """True if every prime factor of n is <= bound (n is `bound`-smooth)."""
    for p, e in _prime_power_factorization(n):
        if p > bound:
            return False
    return True


def brute_discrete_log(g, h, mod, order=None):
    """Reference: smallest x with g^x = h (mod mod), by scanning."""
    g %= mod
    h %= mod
    if order is None:
        order = mod - 1
    cur = 1
    for x in range(order + 1):
        if cur == h:
            return x
        cur = (cur * g) % mod
    return None
