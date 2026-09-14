"""Pratt certificates: a short, independently-checkable PROOF that a number is prime.

Testing primality is one thing; PROVING it to a skeptic is another. Pratt's 1975 certificate (which put
PRIMES in the complexity class NP) is a recursive proof that a number n is prime, small enough to write
down and fast to verify -- polynomially many multiplications. It rests on the LUCAS PRIMALITY TEST, a
converse of Fermat's little theorem: n is prime if and only if there exists a WITNESS a (a primitive
root) such that

    a^{n-1} == 1 (mod n),   and   a^{(n-1)/q} != 1 (mod n) for every prime q dividing n-1.

The first condition is Fermat's congruence; the second says a has multiplicative order exactly n-1, so
the group (Z/nZ)^* has n-1 elements, which forces n prime. The catch is the phrase "every prime q
dividing n-1" -- to trust the proof you must know those q are themselves prime, so the certificate
RECURSES: it lists the factorization of n-1, and for each prime factor it attaches that factor's own
Pratt certificate, all the way down to the base case 2. The whole tree has only O((log n)^2) nodes.

This module builds a Pratt certificate by factoring n-1 (via the repo's elliptic-curve factorizer),
searching for a witness, and recursing on the prime factors; and it independently VERIFIES a certificate
by re-checking every congruence and every claimed factorization from the leaves up. It is validated: a
certificate is produced for every prime and for no composite; the verifier accepts genuine certificates
and rejects tampered ones (wrong witness, missing factor, composite masquerading as prime); the witness
really is a primitive root; the certificate tree bottoms out at 2; and building-then-verifying
round-trips for many primes including large ones. Reuses the repo's Lenstra-ECM factorizer and
Baillie-PSW primality test. Pure stdlib; the primality-proof companion to the AKS, Baillie-PSW, and
Lucas-Lehmer tools."""

from __future__ import annotations

from lenstra_ecm import factorize as _factorize
from baillie_psw import is_prime as _is_prime


def _distinct_prime_factors(m):
    """The set of distinct prime factors of m (via the ECM factorizer)."""
    return sorted(set(_factorize(m)))


def _find_witness(n, factors):
    """Find a primitive root a mod n: a^{n-1}==1 and a^{(n-1)/q}!=1 for each prime q | n-1."""
    for a in range(2, n):
        if pow(a, n - 1, n) != 1:
            continue
        if all(pow(a, (n - 1) // q, n) != 1 for q in factors):
            return a
    return None


def build(n):
    """Build a Pratt certificate for n. Returns a nested structure, or None if n is composite.

    Certificate = (n, witness, [(q, cert_q), ...]) where cert_q is the Pratt certificate of prime q;
    the base case is (2, None, [])."""
    if n < 2:
        return None
    if n == 2:
        return (2, None, [])
    if n % 2 == 0:
        return None                              # even > 2 is composite
    if not _is_prime(n):
        return None
    factors = _distinct_prime_factors(n - 1)
    a = _find_witness(n, factors)
    if a is None:
        return None
    sub = []
    for q in factors:
        cq = build(q)
        if cq is None:
            return None
        sub.append((q, cq))
    return (n, a, sub)


def verify(cert, n=None):
    """Independently verify a Pratt certificate. Returns True iff it proves its number prime."""
    if cert is None:
        return False
    num, a, sub = cert
    if n is not None and num != n:
        return False
    if num == 2:
        return a is None and sub == []
    if num < 2 or num % 2 == 0:
        return False
    # 1. Fermat congruence a^{num-1} == 1 (mod num)
    if pow(a, num - 1, num) != 1:
        return False
    # 2. the listed factors must multiply (with multiplicity) to num - 1, and each be prime-certified
    #    reconstruct num-1 from the distinct primes with their full multiplicity
    m = num - 1
    primes = []
    for q, cq in sub:
        if not verify(cq, q):                    # recursively check q is prime
            return False
        primes.append(q)
        # a^{(num-1)/q} != 1 (mod num): a has order not dividing (num-1)/q
        if pow(a, (num - 1) // q, num) == 1:
            return False
    # 3. the distinct primes must be exactly the prime factors of num-1
    #    (divide out all of them fully; the result must be 1)
    rem = m
    for q in primes:
        while rem % q == 0:
            rem //= q
    if rem != 1:
        return False
    # every prime factor of num-1 is covered and certified, and a is a primitive root -> num is prime
    return True


def certificate_size(cert):
    """Number of nodes in the certificate tree (a proxy for proof length)."""
    if cert is None:
        return 0
    _, _, sub = cert
    return 1 + sum(certificate_size(cq) for _, cq in sub)


def witness(n):
    """The primitive-root witness for prime n (the top-level certificate witness), or None."""
    cert = build(n)
    return cert[1] if cert else None
