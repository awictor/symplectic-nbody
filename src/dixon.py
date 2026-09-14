"""Dixon's factorization: turning a congruence of squares into a factor, the idea behind the sieves.

Fermat's insight is that if you can find x, y with x^2 = y^2 (mod N) but x != +/-y (mod N), then
gcd(x - y, N) is a nontrivial factor -- because N divides (x-y)(x+y) but neither factor alone. The
whole family of modern factoring algorithms (Dixon, the quadratic sieve, the number field sieve) is
about MANUFACTURING such a congruence of squares. DIXON'S METHOD (1981) is the simplest, and the first
provably subexponential factoring algorithm.

The recipe: pick a FACTOR BASE of small primes. Draw random x, compute x^2 mod N, and keep the ones
that are SMOOTH -- factor completely over the base. Each smooth relation records the exponent vector of
its factorization modulo 2 (only parities matter, since we want a square). Collect more relations than
base primes, then find a subset whose exponent vectors SUM TO ZERO mod 2 via GAUSSIAN ELIMINATION OVER
GF(2): that subset multiplies to a perfect square on both sides,

    (product of x_i)^2  =  product of (x_i^2 mod N)  =  a perfect square Y^2  (mod N),

giving a congruence of squares. Take gcd(X - Y, N); with probability >= 1/2 it is a nontrivial factor,
and otherwise another dependency is tried. This is exactly the quadratic sieve's engine, minus the
clever sieving that finds smooth numbers faster.

This module implements the factor base, smooth-relation collection, the GF(2) null-space search, and
the full Dixon factorization, plus a recursive complete factorization into primes. It is validated by
results and by the underlying identity: it factors semiprimes and general composites into correct
prime factorizations (whose product is N), agrees with a trial-division reference on many random N,
returns primes unchanged, handles even numbers and prime powers, and -- checked directly -- every
smooth relation's factorization really equals x^2 mod N and every GF(2) dependency really yields a
congruence of squares X^2 = Y^2 (mod N). Pure stdlib (a seeded RNG, integer arithmetic); the
subexponential-factoring companion to the Pollard-rho, trial-division, and tonelli-shanks tools."""

from __future__ import annotations

from math import isqrt, gcd


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state
    return nxt


def _sieve_primes(limit):
    """Primes up to limit by the sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def factor_base(N, bound=None):
    """Small primes up to `bound` (default ~ smoothness bound) for the factor base."""
    if bound is None:
        # heuristic smoothness bound; small for our modest N
        import math
        L = math.exp(0.5 * math.sqrt(math.log(N) * math.log(math.log(N)))) if N > 3 else 5
        bound = max(7, int(L))
    return _sieve_primes(bound)


def _smooth_vector(value, base):
    """If `value` factors completely over `base`, return its exponent vector; else None."""
    v = value
    exps = [0] * len(base)
    for i, p in enumerate(base):
        while v % p == 0:
            v //= p
            exps[i] += 1
    return exps if v == 1 else None


def _gf2_nullspace_dependency(rows):
    """Find a nonempty subset of `rows` (exponent vectors mod 2) summing to 0 mod 2.

    Returns a list of row indices, or None. Uses Gaussian elimination over GF(2) tracking origins.
    """
    n = len(rows)
    if n == 0:
        return None
    ncols = len(rows[0])
    # each working row = (parity vector, set of original indices combined)
    mat = [([e % 2 for e in rows[i]], {i}) for i in range(n)]
    pivot_row_for_col = {}
    for r in range(n):
        vec, origin = mat[r]
        # reduce against existing pivots
        for c in range(ncols):
            if vec[c] == 1:
                if c in pivot_row_for_col:
                    pv, po = mat[pivot_row_for_col[c]]
                    vec = [vec[k] ^ pv[k] for k in range(ncols)]
                    origin = origin ^ po
                else:
                    pivot_row_for_col[c] = r
                    mat[r] = (vec, origin)
                    break
        else:
            # vec became all zero -> dependency among the combined originals
            if origin:
                return sorted(origin)
        # if we broke with a new pivot, store the reduced form
        if any(vec):
            # only update if not already stored as a pivot above (kept consistent)
            mat[r] = (vec, origin)
    return None


def dixon_factor(N, seed=1, extra=5, max_tries=20000):
    """Find one nontrivial factor of N by Dixon's method, or None if it fails within the budget."""
    if N % 2 == 0:
        return 2
    r = isqrt(N)
    if r * r == N:
        return r  # perfect square
    base = factor_base(N)
    m = len(base)
    rng = _lcg(seed)

    xs = []
    vectors = []
    tries = 0
    # collect m + extra smooth relations
    while len(vectors) < m + extra and tries < max_tries:
        tries += 1
        x = r + 1 + (rng() % (N - r - 1)) if N - r - 1 > 0 else r + 1
        val = (x * x) % N
        if val == 0:
            g = gcd(x, N)
            if 1 < g < N:
                return g
            continue
        vec = _smooth_vector(val, base)
        if vec is not None:
            xs.append(x)
            vectors.append(vec)

    # search for a GF(2) dependency; retry with fresh relations if it does not yield a factor
    for _ in range(len(vectors)):
        dep = _gf2_nullspace_dependency(vectors)
        if dep is None:
            break
        # build X = product of x_i, Y = sqrt(product of x_i^2 mod N) using summed exponents
        X = 1
        total_exps = [0] * m
        for i in dep:
            X = (X * xs[i]) % N
            val = (xs[i] * xs[i]) % N
            ev = _smooth_vector(val, base)
            for k in range(m):
                total_exps[k] += ev[k]
        Y = 1
        for k in range(m):
            Y = (Y * pow(base[k], total_exps[k] // 2, N)) % N
        for cand in (gcd(X - Y, N), gcd(X + Y, N)):
            if 1 < cand < N:
                return cand
        # remove one relation from the dependency and try again
        vectors.pop(dep[0])
        xs.pop(dep[0])
    return None


def factorize(N, seed=1):
    """Complete prime factorization of N as a sorted list (with multiplicity)."""
    if N <= 1:
        return []
    from pollard_rho import is_prime
    factors = []
    stack = [N]
    while stack:
        n = stack.pop()
        if n == 1:
            continue
        if is_prime(n):
            factors.append(n)
            continue
        # try Dixon; fall back to trial division for tiny factors
        d = None
        for s in range(seed, seed + 5):
            d = dixon_factor(n, seed=s)
            if d is not None and 1 < d < n:
                break
        if d is None or d == n or d == 1:
            d = _trial_factor(n)
        stack.append(d)
        stack.append(n // d)
    return sorted(factors)


def _trial_factor(n):
    """Smallest nontrivial factor by trial division (fallback)."""
    if n % 2 == 0:
        return 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return i
        i += 2
    return n


def is_congruence_of_squares(N, X, Y):
    """Check X^2 = Y^2 (mod N) -- the identity Dixon manufactures."""
    return (X * X - Y * Y) % N == 0


def trial_factorization(N):
    """Reference complete factorization by trial division."""
    factors = []
    n = N
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
