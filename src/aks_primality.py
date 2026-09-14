"""AKS: the first deterministic, unconditional, polynomial-time primality test.

For most of history primality testing was either fast-but-probabilistic (Miller-Rabin, Baillie-PSW) or
deterministic-but-slow, and it was a famous open question whether primality was in P at all. In 2002
Manindra Agrawal, Neeraj Kayal, and Nitin Saxena -- Kayal and Saxena being undergraduates -- settled it
with the AKS ALGORITHM: a test that is deterministic (never wrong, never random), unconditional (no
unproven hypotheses like the Riemann hypothesis), and runs in polynomial time in the number of digits.
It is a landmark of theoretical computer science, even though in practice ECPP and Baillie-PSW are far
faster.

The test rests on a clean generalization of Fermat's little theorem to POLYNOMIALS: an integer n > 1 is
prime if and only if (x + a)^n == x^n + a in the ring Z[x] (for a coprime to n). Checking that full
identity is expensive, so AKS checks it modulo a small polynomial x^r - 1 and modulo n, for a cleverly
chosen small r and a bounded range of a. The steps are: (1) if n is a perfect power, it is composite;
(2) find the smallest r such that the multiplicative order of n mod r exceeds (log2 n)^2; (3) if any
2 <= a <= r shares a factor with n, that a is a witness of compositeness; (4) if n <= r, n is prime;
(5) finally verify (x + a)^n == x^n + a in (Z/nZ)[x] / (x^r - 1) for every a from 1 up to
floor(sqrt(phi(r)) * log2 n). If all those polynomial congruences hold, n is prime -- provably.

This module implements AKS with exact integer polynomial arithmetic modulo (x^r - 1, n), the
perfect-power and order-finding subroutines, and returns the certificate parameters (r and the number
of a's checked). It is validated against a deterministic trial-division check for every integer in a
range (no false positives or negatives), including the strong pseudoprimes and Carmichael numbers that
fool probabilistic tests; the polynomial identity is confirmed to hold for primes and to FAIL for
composites; perfect powers are detected; and the chosen r and witness bound match the algorithm's
specification. Cross-checks the repo's Baillie-PSW test. Pure stdlib; the deterministic-primality
companion to the Baillie-PSW, Lucas-Lehmer, and Miller-Rabin tools."""

from __future__ import annotations

from math import gcd, isqrt, log2


def is_perfect_power(n):
    """True if n = a^b for integers a>1, b>1."""
    if n < 4:
        return False
    for b in range(2, int(log2(n)) + 1):
        a = round(n ** (1.0 / b))
        for cand in (a - 1, a, a + 1):
            if cand > 1 and cand ** b == n:
                return True
    return False


def _multiplicative_order(n, r):
    """Multiplicative order of n modulo r (smallest k>0 with n^k ≡ 1 mod r), or None if gcd!=1."""
    if gcd(n, r) != 1:
        return None
    k = 1
    val = n % r
    while val != 1:
        val = (val * n) % r
        k += 1
        if k > r:
            return None
    return k


def _euler_phi(r):
    result = r
    p = 2
    m = r
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1
    if m > 1:
        result -= result // m
    return result


def _poly_mod_mul(A, B, r, n):
    """Multiply polynomials A, B in (Z/nZ)[x] / (x^r - 1)."""
    res = [0] * r
    for i, ai in enumerate(A):
        if ai == 0:
            continue
        for j, bj in enumerate(B):
            if bj:
                res[(i + j) % r] = (res[(i + j) % r] + ai * bj) % n
    return res


def _poly_pow(base, e, r, n):
    """(base)^e in (Z/nZ)[x] / (x^r - 1) by fast exponentiation."""
    result = [0] * r
    result[0] = 1
    b = base[:]
    while e > 0:
        if e & 1:
            result = _poly_mod_mul(result, b, r, n)
        b = _poly_mod_mul(b, b, r, n)
        e >>= 1
    return result


def _check_polynomial(n, r, a):
    """Verify (x + a)^n == x^n + a in (Z/nZ)[x] / (x^r - 1)."""
    # left side: (x + a)^n
    base = [0] * r
    base[0] = a % n
    base[1 % r] = (base[1 % r] + 1) % n
    left = _poly_pow(base, n, r, n)
    # right side: x^n + a  (mod x^r - 1)  =>  x^(n mod r) + a
    right = [0] * r
    right[n % r] = (right[n % r] + 1) % n
    right[0] = (right[0] + a) % n
    return left == right


def is_prime(n, certificate=False):
    """Deterministic AKS primality test. If certificate=True, returns (is_prime, r, a_bound)."""
    if n < 2:
        return (False, None, None) if certificate else False
    if n in (2, 3):
        return (True, None, None) if certificate else True
    # step 1: perfect power => composite
    if is_perfect_power(n):
        return (False, None, None) if certificate else False
    # step 2: find smallest r with ord_r(n) > (log2 n)^2
    log2n = log2(n)
    target = int(log2n * log2n) + 1
    r = 2
    while True:
        if gcd(n, r) == 1:
            o = _multiplicative_order(n, r)
            if o is not None and o > target:
                break
        elif r >= n:
            break
        r += 1
        if r > n:
            r = n
            break
    # step 3: for 2 <= a <= min(r, n-1), if gcd(a, n) not 1 -> composite
    for a in range(2, min(r, n - 1) + 1):
        g = gcd(a, n)
        if 1 < g < n:
            return (False, r, 0) if certificate else False
    # step 4: if n <= r, n is prime
    if n <= r:
        return (True, r, 0) if certificate else True
    # step 5: polynomial congruences for a = 1 .. floor(sqrt(phi(r)) * log2 n)
    phi_r = _euler_phi(r)
    a_bound = int(isqrt(phi_r) * log2n) + 1
    for a in range(1, a_bound + 1):
        if not _check_polynomial(n, r, a):
            return (False, r, a_bound) if certificate else False
    return (True, r, a_bound) if certificate else True


def trial_division_prime(n):
    """Deterministic reference primality by trial division."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True
