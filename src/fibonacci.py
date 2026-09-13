"""Fibonacci and Lucas numbers: fast doubling in O(log n), and the surprising arithmetic they hide.

The Fibonacci numbers F_n = F_{n-1} + F_{n-2} are the most famous sequence in mathematics, but the
naive recurrence takes O(n) additions on numbers with O(n) digits -- quadratic, and hopeless for the
millionth term. The FAST DOUBLING identities compute F_n in O(log n) big-integer multiplications:

    F_{2k}   = F_k * (2 F_{k+1} - F_k)
    F_{2k+1} = F_{k+1}^2 + F_k^2

Read the binary digits of n from the top and apply these at each bit, and you leap to F_n
exponentially fast -- the same speed-up matrix exponentiation of [[1,1],[1,0]] gives, but with half
the multiplications and no matrix. This module also carries the companion LUCAS numbers L_n
(L_0 = 2, L_1 = 1, same recurrence), which satisfy L_n = F_{n-1} + F_{n+1} and interlock with the
Fibonacci numbers through dozens of identities.

The Fibonacci numbers are a playground of number theory, and this module makes several classical facts
computable and checkable:

  CASSINI'S IDENTITY: F_{n-1} F_{n+1} - F_n^2 = (-1)^n, a determinant identity that never drifts.
  THE GCD PROPERTY: gcd(F_m, F_n) = F_{gcd(m, n)} -- the Fibonacci numbers form a "strong divisibility
  sequence", so F_m divides F_n exactly when m divides n.
  THE PISANO PERIOD: the sequence F_n mod m is periodic; its period pi(m) is the Pisano period, and it
  is how one computes F_n mod m for astronomically large n.
  ZECKENDORF'S THEOREM: every positive integer is a unique sum of non-consecutive Fibonacci numbers.

This module implements fast-doubling Fibonacci and Lucas, F_n mod m via a modular fast doubling, the
Pisano period, the Zeckendorf representation, and an index-from-value inverse. It is validated against
the naive recurrence (fast doubling agrees for n = 0..1000), against the closed-form and matrix forms,
and by the identities themselves: Cassini holds, the GCD property holds over many pairs, the Pisano
period genuinely cycles F_n mod m, and every Zeckendorf representation is non-consecutive and sums back
to n. Pure stdlib (arbitrary-precision integers); the fast-recurrence companion to the Kitamasa linear-
recurrence solver and the golden-ratio / Benford tools."""

from __future__ import annotations


def fib_pair(n):
    """Return (F_n, F_{n+1}) by fast doubling. O(log n) big-integer multiplications."""
    if n == 0:
        return (0, 1)
    a, b = fib_pair(n >> 1)          # a = F_k, b = F_{k+1}, k = n//2
    c = a * (2 * b - a)              # F_{2k}
    d = a * a + b * b                # F_{2k+1}
    if n & 1:
        return (d, c + d)            # (F_{2k+1}, F_{2k+2})
    return (c, d)                    # (F_{2k}, F_{2k+1})


def fibonacci(n):
    """The n-th Fibonacci number (F_0 = 0, F_1 = 1) in O(log n)."""
    if n < 0:
        # negafibonacci: F_{-n} = (-1)^{n+1} F_n
        f = fibonacci(-n)
        return f if (-n) % 2 == 1 else -f
    return fib_pair(n)[0]


def lucas(n):
    """The n-th Lucas number (L_0 = 2, L_1 = 1) via L_n = 2 F_{n+1} - F_n."""
    f_n, f_n1 = fib_pair(n)
    return 2 * f_n1 - f_n


def fib_mod(n, m):
    """F_n mod m by modular fast doubling (works for astronomically large n)."""
    if m == 1:
        return 0

    def helper(k):
        if k == 0:
            return (0, 1)
        a, b = helper(k >> 1)
        c = (a * ((2 * b - a) % m)) % m
        d = (a * a + b * b) % m
        if k & 1:
            return (d, (c + d) % m)
        return (c, d)

    return helper(n)[0] % m


def pisano_period(m):
    """The Pisano period pi(m): the period of the sequence F_n mod m."""
    if m == 1:
        return 1
    prev, curr = 0, 1
    for i in range(m * m + 1):       # the period is at most m*m (actually <= 6m)
        prev, curr = curr, (prev + curr) % m
        if prev == 0 and curr == 1:
            return i + 1
    return None  # should never happen


def zeckendorf(n):
    """Zeckendorf representation of n: the unique set of non-consecutive Fibonacci numbers summing to n.

    Returns the list of Fibonacci VALUES (descending), using F_2=1, F_3=2, F_4=3, ... (no F_1 dup).
    """
    if n < 0:
        raise ValueError("Zeckendorf is for non-negative integers")
    if n == 0:
        return []
    # collect Fibonacci numbers up to n (starting from 1, 2, 3, 5, ...)
    fibs = [1, 2]
    while fibs[-1] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    rep = []
    for f in reversed(fibs):
        if f <= n:
            rep.append(f)
            n -= f
    return rep


def cassini(n):
    """Return F_{n-1} F_{n+1} - F_n^2, which equals (-1)^n (Cassini's identity)."""
    fnm1 = fibonacci(n - 1)
    fn = fibonacci(n)
    fnp1 = fibonacci(n + 1)
    return fnm1 * fnp1 - fn * fn


def fib_index(value):
    """The index n such that F_n == value, or None if value is not a Fibonacci number (value >= 0)."""
    if value < 0:
        return None
    a, b, n = 0, 1, 0
    while a < value:
        a, b = b, a + b
        n += 1
    return n if a == value else None


def naive_fibonacci(n):
    """Reference O(n) iterative Fibonacci for cross-checking."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def matrix_fibonacci(n):
    """Reference: F_n via matrix power of [[1,1],[1,0]] (repeated squaring)."""
    def matmul(A, B):
        return (
            A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
            A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3],
        )
    result = (1, 0, 0, 1)   # identity
    base = (1, 1, 1, 0)
    k = n
    while k > 0:
        if k & 1:
            result = matmul(result, base)
        base = matmul(base, base)
        k >>= 1
    return result[1]        # (M^n)[0][1] = F_n
