"""Dirichlet convolution: the multiplication that makes arithmetic functions an algebra.

Number theory is full of functions defined on the integers -- the Euler totient phi, the Mobius
function mu, the divisor-count d and divisor-sum sigma, the constant 1, the identity id(n) = n. They
seem unrelated until you equip them with the right product: the DIRICHLET CONVOLUTION

    (f * g)(n) = sum over divisors d of n of f(d) g(n/d).

Under this product the arithmetic functions form a commutative ring whose identity is the function
epsilon(n) = [n == 1]. Suddenly the classical identities become one-line algebra:

    mu * 1 = epsilon          (the defining property of the Mobius function)
    phi * 1 = id              (sum of phi over divisors is n)
    1 * 1 = d                 (the number-of-divisors function)
    id * 1 = sigma            (the sum-of-divisors function)
    mu * id = phi             (a form of Mobius inversion)

MOBIUS INVERSION is just the statement that 1 is invertible with inverse mu: if F = f * 1 (F is the
"divisor sum" of f), then f = F * mu. This is the discrete analogue of the fundamental theorem of
calculus -- it recovers a function from its cumulative divisor sums -- and it underlies inclusion-
exclusion, counting primitive objects (necklaces, Lyndon words), and evaluating sums over coprime
pairs.

This module represents an arithmetic function as its values on 1..N, implements the Dirichlet
convolution and Dirichlet inverse, and provides the standard functions (epsilon, one, id, mu, phi, d,
sigma_k) built from a linear sieve. It is validated by the identities themselves: mu * 1 = epsilon,
phi * 1 = id, 1 * 1 = d, id * 1 = sigma, mu * id = phi, convolution is commutative and associative,
epsilon is the identity, the Dirichlet inverse of 1 is mu, and Mobius inversion round-trips
(f * 1) * mu = f -- all checked exactly against independently sieved values. Pure stdlib; the
arithmetic-function-algebra companion to the linear sieve and the multiplicative-function tools."""

from __future__ import annotations

from linear_sieve import linear_sieve


def dirichlet_convolution(f, g, N):
    """(f * g)(n) = sum_{d | n} f(d) g(n/d) for n = 1..N. f, g are 1-indexed lists (index 0 unused)."""
    h = [0] * (N + 1)
    for d in range(1, N + 1):
        fd = f[d]
        if fd == 0:
            continue
        for m in range(d, N + 1, d):
            h[m] += fd * g[m // d]
    return h


def dirichlet_inverse(f, N):
    """The Dirichlet inverse f^{-1} with f * f^{-1} = epsilon. Requires f[1] != 0.

    Computed by the recurrence f^{-1}(1) = 1/f(1), and for n > 1
    f^{-1}(n) = -1/f(1) * sum_{d|n, d<n} f(n/d) f^{-1}(d).
    """
    if f[1] == 0:
        raise ValueError("Dirichlet inverse requires f(1) != 0")
    from fractions import Fraction
    inv = [Fraction(0)] * (N + 1)
    inv[1] = Fraction(1, f[1])
    for n in range(2, N + 1):
        s = Fraction(0)
        d = 1
        while d < n:
            if n % d == 0:
                s += f[n // d] * inv[d]
            d += 1
        inv[n] = -s / f[1]
    return inv


# ---- standard arithmetic functions on 1..N -----------------------------------------------------

def epsilon(N):
    """epsilon(n) = [n == 1], the Dirichlet identity."""
    e = [0] * (N + 1)
    if N >= 1:
        e[1] = 1
    return e


def one(N):
    """The constant function 1."""
    return [0] + [1] * N


def identity(N):
    """id(n) = n."""
    return list(range(N + 1))


def mobius(N):
    """The Mobius function mu(n) via the linear sieve."""
    return linear_sieve(N)["mu"]


def totient(N):
    """The Euler totient phi(n) via the linear sieve."""
    return linear_sieve(N)["phi"]


def divisor_count(N):
    """d(n) = number of divisors, as 1 * 1."""
    return dirichlet_convolution(one(N), one(N), N)


def divisor_sum(N, k=1):
    """sigma_k(n) = sum of d^k over divisors d of n. sigma_1 = sigma, sigma_0 = d."""
    powk = [0] + [d ** k for d in range(1, N + 1)]
    return dirichlet_convolution(powk, one(N), N)


def mobius_inversion(F, N):
    """Given F = f * 1 (divisor sums of f), recover f = F * mu."""
    return dirichlet_convolution(F, mobius(N), N)


def divisor_sum_transform(f, N):
    """F(n) = sum_{d | n} f(d), i.e. F = f * 1."""
    return dirichlet_convolution(f, one(N), N)


# ---- brute-force references -------------------------------------------------------------------

def brute_divisor_count(n):
    return sum(1 for d in range(1, n + 1) if n % d == 0)


def brute_divisor_sum(n, k=1):
    return sum(d ** k for d in range(1, n + 1) if n % d == 0)
