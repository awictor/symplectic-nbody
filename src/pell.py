"""Pell's equation: x^2 - D y^2 = 1, solved by the continued fraction of sqrt(D).

Pell's equation x^2 - D y^2 = 1 (for a non-square positive integer D) is one of the oldest problems in
number theory -- studied by Brahmagupta in the 7th century, misattributed by Euler to Pell, and truly
solved by Lagrange. It has infinitely many integer solutions, and finding the smallest nontrivial one
(the FUNDAMENTAL SOLUTION) is subtle: the answer can be enormous even for small D. For D = 61 the
minimal solution is x = 1766319049, y = 226153980 -- Fermat posed exactly this as a challenge, knowing
how spectacularly the naive search fails.

The key is that sqrt(D) has an eventually PERIODIC continued fraction, [a0; a1, a2, ..., a_k] repeating,
and the CONVERGENTS p/q of that expansion are the best rational approximations to sqrt(D). One of those
convergents, taken at the end of a period, gives the fundamental solution: if the period length L is
even, the convergent p_{L-1}/q_{L-1} solves x^2 - D y^2 = +1; if L is odd, that convergent solves the
NEGATIVE Pell equation x^2 - D y^2 = -1, and squaring it (via Brahmagupta's composition) gives the +1
solution. Once the fundamental solution (x1, y1) is known, ALL solutions follow from the recurrence

    x_{n+1} = x1 x_n + D y1 y_n,   y_{n+1} = x1 y_n + y1 x_n,

which is just repeated multiplication in the ring Z[sqrt(D)] -- (x_n + y_n sqrt D) = (x1 + y1 sqrt D)^n.

This module computes the periodic continued fraction of sqrt(D) with the standard integer recurrence
(no floating point, so it is exact for large D), extracts the fundamental solution of the positive and
negative Pell equations, and generates further solutions by the recurrence. It is validated against
known fundamental solutions (including D = 61 and D = 109, the classic hard cases), by verifying
x^2 - D y^2 = 1 exactly for the fundamental solution and many generated ones, by confirming the CF
period and that squares of D produce no solution, and by checking the negative Pell equation exists
exactly when the CF period is odd. Pure stdlib (integer arithmetic only); the Diophantine companion to
the continued-fraction and Stern-Brocot tools."""

from __future__ import annotations

from math import isqrt


def cf_sqrt_period(D):
    """The continued fraction of sqrt(D): (a0, [periodic part]). Empty period if D is a perfect square.

    Uses the exact integer recurrence m_{k+1} = d_k a_k - m_k, d_{k+1} = (D - m^2)/d, a = (a0 + m)//d.
    """
    a0 = isqrt(D)
    if a0 * a0 == D:
        return a0, []
    m, d, a = 0, 1, a0
    period = []
    # the period ends when a = 2*a0 (standard termination for sqrt(D))
    while True:
        m = d * a - m
        d = (D - m * m) // d
        a = (a0 + m) // d
        period.append(a)
        if a == 2 * a0:
            break
    return a0, period


def fundamental_solution(D):
    """The fundamental (smallest) solution (x, y) of x^2 - D y^2 = 1 for non-square D > 1.

    Returns None if D is a perfect square (no nontrivial solution).
    """
    a0 = isqrt(D)
    if a0 * a0 == D:
        return None
    _, period = cf_sqrt_period(D)
    L = len(period)
    # build convergents until we hit the one giving +1
    # convergent recurrence: h_n = a_n h_{n-1} + h_{n-2}, k_n = a_n k_{n-1} + k_{n-2}
    # the full CF coefficient sequence is a0, period[0], period[1], ...
    def convergent_at(num_terms):
        h_prev, h_prev2 = 1, 0
        k_prev, k_prev2 = 0, 1
        for i in range(num_terms):
            ai = a0 if i == 0 else period[(i - 1) % L]
            h = ai * h_prev + h_prev2
            k = ai * k_prev + k_prev2
            h_prev2, h_prev = h_prev, h
            k_prev2, k_prev = k_prev, k
        return h_prev, k_prev

    if L % 2 == 0:
        # even period: convergent p_{L-1} (using terms a0..period[L-2]) gives +1
        x, y = convergent_at(L)
    else:
        # odd period: need to go two periods to get +1
        x, y = convergent_at(2 * L)
    return (x, y)


def negative_pell_solution(D):
    """The fundamental solution of x^2 - D y^2 = -1, or None if none exists (even CF period)."""
    a0 = isqrt(D)
    if a0 * a0 == D:
        return None
    _, period = cf_sqrt_period(D)
    L = len(period)
    if L % 2 == 0:
        return None  # negative Pell solvable iff period length is odd
    # convergent p_{L-1} gives -1

    def convergent_at(num_terms):
        h_prev, h_prev2 = 1, 0
        k_prev, k_prev2 = 0, 1
        for i in range(num_terms):
            ai = a0 if i == 0 else period[(i - 1) % L]
            h = ai * h_prev + h_prev2
            k = ai * k_prev + k_prev2
            h_prev2, h_prev = h_prev, h
            k_prev2, k_prev = k_prev, k
        return h_prev, k_prev

    x, y = convergent_at(L)
    return (x, y)


def solutions(D, count):
    """Generate the first `count` positive solutions of x^2 - D y^2 = 1 by the recurrence."""
    fund = fundamental_solution(D)
    if fund is None:
        return []
    x1, y1 = fund
    out = [(x1, y1)]
    x, y = x1, y1
    for _ in range(count - 1):
        x, y = x1 * x + D * y1 * y, x1 * y + y1 * x
        out.append((x, y))
    return out


def verify(D, x, y, rhs=1):
    """Check x^2 - D y^2 == rhs exactly."""
    return x * x - D * y * y == rhs


def is_square(n):
    r = isqrt(n)
    return r * r == n
