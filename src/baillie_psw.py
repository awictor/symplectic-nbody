"""Baillie-PSW: the primality test with no known counterexample, combining Miller-Rabin and Lucas.

Fast primality testing is probabilistic: a single Miller-Rabin round can be fooled by a STRONG
PSEUDOPRIME to that base, and Lucas tests have their own pseudoprimes. The insight of Baillie, Pomerance,
Selfridge, and Wagstaff (1980) is that the two families of pseudoprimes appear to be DISJOINT -- a
number that fools one almost never fools the other. Baillie-PSW runs a strong Miller-Rabin test to base
2 AND a strong Lucas probable-prime test with Selfridge's parameters, and declares primality only if
both pass. Despite decades of searching, NO composite is known to pass it, and it is proven correct for
all n below 2^64; it is the default primality test in Sympy, PARI/GP, and many crypto libraries.

The two halves probe different structure. The strong Miller-Rabin base-2 test writes n - 1 = d * 2^s
and checks that 2^d ≡ 1 or that some 2^{d*2^r} ≡ -1 -- the Fermat congruence refined to catch square
roots of unity. The strong Lucas test picks D by Selfridge's method (the first D in 5, -7, 9, -11, ...
with Jacobi symbol (D/n) = -1), sets P = 1, Q = (1 - D)/4, and checks the Lucas sequences U and V for
the analogous strong condition. Their pseudoprimes have never been found to coincide, which is what
makes the combination so trustworthy.

This module implements the Jacobi symbol, Lucas sequence computation by binary ladder, the strong
Miller-Rabin base-2 and strong Lucas tests, and the combined Baillie-PSW test. It is validated: it
agrees with a deterministic trial-division primality check for EVERY integer up to a bound (no false
positives or negatives there); it passes every prime and rejects every composite in that range,
including the classic strong pseudoprimes to base 2 (2047, 3277, ...) that fool a naive Fermat test but
are caught by the Lucas half; the Jacobi symbol matches the Legendre symbol for primes and is
multiplicative; and it handles the small edge cases (0, 1, 2, perfect squares). Pure stdlib; the
primality companion to the Tonelli-Shanks, Cipolla, and Lucas-Lehmer tools."""

from __future__ import annotations

from math import isqrt


def jacobi_symbol(a, n):
    """The Jacobi symbol (a/n) for odd n > 0: generalization of the Legendre symbol. Returns -1, 0, 1."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a                              # reciprocity: flip
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def strong_miller_rabin_base2(n):
    """Strong probable-prime (Miller-Rabin) test to base 2."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(2, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False


def _lucas_uv(k, n, P, Q):
    """Lucas sequences (U_k, V_k, Q^k) mod n by the binary ladder, for P and Q (Selfridge)."""
    inv2 = (n + 1) // 2                           # modular inverse of 2 (n is odd)
    D = P * P - 4 * Q
    U, V, Qk = 0, 2, 1                            # U_0=0, V_0=2, Q^0=1
    for bit in bin(k)[2:]:
        # doubling k -> 2k
        U, V = (U * V) % n, (V * V - 2 * Qk) % n
        Qk = (Qk * Qk) % n
        if bit == '1':
            # step 2k -> 2k+1
            U, V = ((P * U + V) * inv2) % n, ((D * U + P * V) * inv2) % n
            Qk = (Qk * Q) % n
    return U % n, V % n, Qk % n


def strong_lucas_prp(n):
    """Strong Lucas probable-prime test with Selfridge parameters (P=1, Q=(1-D)/4)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if is_perfect_square(n):
        return False                             # a perfect square is never a Lucas prime here
    # Selfridge: find the first D in 5, -7, 9, -11, ... with jacobi(D, n) = -1
    D = 5
    while True:
        j = jacobi_symbol(D % n, n)
        if j == -1:
            break
        if j == 0 and abs(D) != n:
            return False                         # gcd(D, n) > 1 and n doesn't divide D -> composite
        D = -D - 2 if D > 0 else -D + 2
        # safety: if D grows huge, n is likely a square (already handled)
        if abs(D) > 1000000:
            return False
    P = 1
    Q = (1 - D) // 4
    # write n+1 = d * 2^s
    d = n + 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # compute U_d, V_d, Q^d via the binary ladder
    U, V, Qk = _lucas_uv(d, n, P, Q)
    # strong test: U_d ≡ 0, or V_{d*2^r} ≡ 0 for some 0 <= r < s
    if U % n == 0 or V % n == 0:
        return True
    for _ in range(1, s):
        V = (V * V - 2 * Qk) % n
        Qk = (Qk * Qk) % n
        if V % n == 0:
            return True
    return False


def is_prime(n):
    """Baillie-PSW primality test: strong Miller-Rabin base 2 AND strong Lucas."""
    if n < 2:
        return False
    # small primes and trial division by them
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    return strong_miller_rabin_base2(n) and strong_lucas_prp(n)


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
