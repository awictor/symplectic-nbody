"""Lucas-Lehmer: the deterministic test that finds the largest known primes.

Every record-breaking largest-known prime for decades has been a MERSENNE PRIME, a prime of the form
M_p = 2^p - 1, and the reason is the LUCAS-LEHMER test: a startlingly simple, deterministic
primality test for exactly these numbers. Where general primality testing needs probabilistic
witnesses or heavy machinery, Lucas-Lehmer decides whether M_p is prime with p - 2 squarings and one
final comparison -- and it is EXACT, not probabilistic. This is the algorithm the GIMPS distributed
project runs on tens of thousands of machines to hunt primes with tens of millions of digits.

The test: for an odd prime p, define the sequence s_0 = 4, s_{k+1} = s_k^2 - 2, all taken modulo
M_p = 2^p - 1. Then

    M_p is prime  if and only if  s_{p-2} = 0 (mod M_p).

That is the entire test. The modular reduction by 2^p - 1 is itself cheap (a shift-and-add trick,
since x mod (2^p - 1) folds the high bits back onto the low ones), which is why the test scales to
astronomical p. The underlying reason it works is the theory of LUCAS SEQUENCES U_n, V_n -- companion
sequences to a quadratic x^2 - Px + Q whose divisibility properties encode primality; the s_k are
essentially the V-sequence of x^2 - 4x + 1 evaluated at the right index.

This module implements the Lucas-Lehmer test with the fast Mersenne reduction, the general Lucas
sequences U_n(P, Q) and V_n(P, Q) by fast doubling, and a Lucas probable-prime test. It is validated
exactly: it identifies the Mersenne-prime exponents p in {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}
and rejects the composite ones (11, 23, 29, 37, ...), agreeing with a trial-division / repo primality
check on 2^p - 1 for all small p; the Lucas sequences satisfy their defining recurrences and the
identity V_n^2 - D U_n^2 = 4 Q^n; and the Mersenne fast reduction matches the ordinary modulo. Pure
stdlib (arbitrary-precision integers); the Mersenne-prime companion to the Fibonacci fast-doubling and
Pollard-rho / Miller-Rabin primality tools."""

from __future__ import annotations


def _mersenne_mod(x, p):
    """x mod (2^p - 1) by folding high bits onto low ones (repeated until < 2^p - 1)."""
    M = (1 << p) - 1
    # fold high bits down while x has more than p bits; when x == M the fold is a fixed point
    # (M & M) + (M >> p) = M, so guard with '>' and normalize the M case separately.
    while x > M:
        x = (x & M) + (x >> p)
    if x == M:
        x = 0
    return x


def lucas_lehmer(p):
    """Lucas-Lehmer test: is M_p = 2^p - 1 prime? p should be an odd prime (M_2 = 3 handled).

    Returns True if 2^p - 1 is prime, False otherwise.
    """
    if p == 2:
        return True  # M_2 = 3 is prime
    if p < 2:
        return False
    M = (1 << p) - 1
    s = 4
    for _ in range(p - 2):
        s = _mersenne_mod(s * s - 2, p)
    return s == 0


def mersenne_prime_exponents(limit):
    """All p <= limit for which M_p = 2^p - 1 is prime (p must itself be prime)."""
    from pollard_rho import is_prime
    out = []
    for p in range(2, limit + 1):
        if not is_prime(p):
            continue  # M_p composite unless p is prime
        if lucas_lehmer(p):
            out.append(p)
    return out


# ---- general Lucas sequences U_n(P, Q), V_n(P, Q) ---------------------------------------------

def D_of(P, Q):
    """The discriminant D = P^2 - 4Q of the Lucas sequence's characteristic x^2 - Px + Q."""
    return P * P - 4 * Q


def lucas_sequence(n, P, Q, m=None):
    """Return (U_n, V_n, Q^n) for the Lucas sequences of x^2 - P x + Q, by fast doubling.

    U_0=0, U_1=1, V_0=2, V_1=P; U_{k+1}=P U_k - Q U_{k-1}, similarly for V. If m is given, all values
    are reduced modulo m. O(log n).
    """
    def reduce(x):
        return x % m if m is not None else x

    # fast doubling for Lucas sequences
    # base: U_0=0, V_0=2, and we build up bit by bit
    U, V, qk = 0, 2, 1  # at index 0
    # process bits of n from most significant
    for bit in bin(n)[2:]:
        # doubling: index d -> 2d
        U2 = reduce(U * V)
        V2 = reduce(V * V - 2 * qk)
        qk2 = reduce(qk * qk)
        U, V, qk = U2, V2, qk2
        if bit == "1":
            # increment: index 2d -> 2d+1 via U_{k+1}=(P U_k + V_k)/2, V_{k+1}=((P^2-4Q)U_k + P V_k)/2.
            # Both numerators are even; divide by 2 exactly (or by the modular inverse of 2).
            newU = P * U + V
            newV = (P * P - 4 * Q) * U + P * V
            U = reduce(newU // 2) if m is None else reduce(_half_mod(newU, m))
            V = reduce(newV // 2) if m is None else reduce(_half_mod(newV, m))
            qk = reduce(qk * Q)
    return U, V, qk


def _half_mod(x, m):
    """x/2 mod m when x is even (or via modular inverse of 2)."""
    if x % 2 == 0:
        return (x // 2) % m
    inv2 = pow(2, -1, m)
    return (x * inv2) % m


def lucas_sequence_naive(n, P, Q):
    """Reference: U_n, V_n by the direct recurrence (for cross-checking)."""
    U0, U1 = 0, 1
    V0, V1 = 2, P
    if n == 0:
        return 0, 2
    for _ in range(n - 1):
        U0, U1 = U1, P * U1 - Q * U0
        V0, V1 = V1, P * V1 - Q * V0
    return U1, V1


def jacobi_symbol(a, n):
    """The Jacobi symbol (a/n) for odd n > 0, in {-1, 0, 1}."""
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def is_lucas_probable_prime(n, P=1, Q=-1):
    """A Lucas probable-prime test: n is a Lucas PRP if U_{n-(D/n)} = 0 mod n (D = P^2-4Q).

    (D/n) is the Jacobi symbol -- for a true prime U_{p-(D/p)} vanishes, whereas a naive U_{n+1} only
    works when (D/n) = -1. Default (P,Q)=(1,-1) is the Fibonacci Lucas sequence; combined with a base-2
    Fermat test this is the strong Baillie-PSW test. Composite Lucas pseudoprimes exist but are rare.
    """
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    D = D_of(P, Q)
    j = jacobi_symbol(D, n)
    idx = n - j  # n - (D/n)
    U, V, qk = lucas_sequence(idx, P, Q, m=n)
    return U % n == 0


def mersenne_reduction_check(x, p):
    """Return whether the fast Mersenne reduction equals the ordinary modulo (for testing)."""
    M = (1 << p) - 1
    return _mersenne_mod(x, p) == x % M
