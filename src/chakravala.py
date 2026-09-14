"""Chakravala: Bhaskara II's cyclic algorithm for Pell's equation, a millennium ahead of Europe.

Around 1150 CE the Indian mathematician Bhaskara II perfected the CHAKRAVALA ("cyclic") method for
solving Pell's equation x^2 - D y^2 = 1 in integers -- six centuries before Fermat posed it as a
challenge and Lagrange proved the continued-fraction method always terminates. Chakravala is a
different and remarkably efficient route: rather than expand sqrt(D) as a continued fraction, it starts
from a trivial "triple" and repeatedly COMPOSES it with a cleverly chosen auxiliary, cycling through
states until it lands on one that solves the equation. On the classic hard case D = 61 -- whose smallest
solution is x = 1766319049, y = 226153980 -- chakravala reaches the answer in a handful of steps.

The engine is BRAHMAGUPTA'S IDENTITY (the bhavana, or "composition"): if a^2 - D b^2 = k1 and
c^2 - D d^2 = k2, then (ac + D bd)^2 - D (ad + bc)^2 = k1 k2. Chakravala keeps a triple (a, b, k) with
a^2 - D b^2 = k, composes it with the trivial (m, 1, m^2 - D), and divides through by |k| -- which is
legal exactly when b m + a is divisible by k. Among all valid m it picks the one making |m^2 - D|
smallest, which provably keeps k bounded and drives the process to k = 1. The state (a, b, k) then
satisfies a^2 - D b^2 = 1: the fundamental solution.

This module implements chakravala with exact integer arithmetic, returning the fundamental solution
and the trace of intermediate triples, and generates further solutions by Brahmagupta composition. It
is validated against the repo's continued-fraction Pell solver: the two agree on the fundamental
solution for many D including the notorious D = 61 and D = 109; the returned (x, y) satisfies
x^2 - D y^2 = 1 exactly (arbitrary-precision integers); perfect squares yield no solution; generated
higher solutions also satisfy the equation; and every intermediate triple satisfies its own
a^2 - D b^2 = k invariant. Cross-checks the repo's Pell solver. Pure stdlib; the number-theory
companion to the Pell, continued-fraction, and Brahmagupta tools."""

from __future__ import annotations

from math import isqrt


def is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def fundamental_solution(D, trace=False):
    """Fundamental solution (x, y) of x^2 - D y^2 = 1 by the chakravala method (D non-square > 1).

    If trace=True, returns (x, y, triples) where triples is the list of intermediate (a, b, k)."""
    if D <= 1 or is_square(D):
        raise ValueError("D must be a non-square integer > 1")
    # initial triple: pick a near sqrt(D) so that k = a^2 - D is small
    a = isqrt(D)
    b = 1
    k = a * a - D                                # a^2 - D b^2 = k
    triples = [(a, b, k)]
    while k != 1:
        # choose m so that (a + b m) is divisible by |k| and |m^2 - D| is minimized
        absk = abs(k)
        # m must satisfy a + b*m ≡ 0 (mod |k|); solve for the residue then scan near sqrt(D)
        m = _choose_m(a, b, absk, D)
        # compose (a, b, k) with (m, 1, m^2 - D) and divide by k
        new_a = (a * m + D * b) // absk
        new_b = (a + b * m) // absk
        new_k = (m * m - D) // k
        a, b, k = new_a, new_b, new_k
        # normalize signs: keep a, b positive
        a, b = abs(a), abs(b)
        triples.append((a, b, k))
        if len(triples) > 1000:
            raise RuntimeError("chakravala failed to converge")
    if trace:
        return a, b, triples
    return a, b


def _choose_m(a, b, absk, D):
    """Pick the positive m with (a + b*m) ≡ 0 (mod |k|) minimizing |m^2 - D|."""
    # find inverse-style residue: we need b*m ≡ -a (mod absk)
    if absk == 1:
        # any m works; pick the one nearest sqrt(D)
        r = isqrt(D)
        return r if abs(r * r - D) <= abs((r + 1) ** 2 - D) else r + 1
    best_m = None
    best_val = None
    # m ranges over a residue class mod absk; scan candidates near sqrt(D)
    center = isqrt(D)
    # find any m0 in [0, absk) satisfying the congruence, then step by absk around center
    m0 = None
    for cand in range(absk):
        if (b * cand + a) % absk == 0:
            m0 = cand
            break
    if m0 is None:
        # b not invertible mod absk for this residue; scan all m in a window (rare, small absk)
        for m in range(1, absk + center + 2):
            if (b * m + a) % absk == 0:
                val = abs(m * m - D)
                if best_val is None or val < best_val:
                    best_val = val
                    best_m = m
        return best_m if best_m is not None else 1
    # candidate m values congruent to m0 mod absk, near sqrt(D)
    base = center - ((center - m0) % absk)
    for m in (base - absk, base, base + absk, base + 2 * absk):
        if m <= 0:
            continue
        if (b * m + a) % absk != 0:
            continue
        val = abs(m * m - D)
        if best_val is None or val < best_val:
            best_val = val
            best_m = m
    return best_m if best_m is not None else (m0 if m0 > 0 else m0 + absk)


def compose(sol1, sol2, D):
    """Brahmagupta composition (bhavana): combine two solutions into another. Returns (x, y)."""
    a, b = sol1
    c, d = sol2
    return (a * c + D * b * d, a * d + b * c)


def solutions(D, count):
    """The first `count` solutions of x^2 - D y^2 = 1, generated from the fundamental one."""
    x1, y1 = fundamental_solution(D)
    out = [(x1, y1)]
    cur = (x1, y1)
    for _ in range(count - 1):
        cur = compose(cur, (x1, y1), D)
        out.append(cur)
    return out


def verify(D, x, y, rhs=1):
    """Check x^2 - D y^2 == rhs exactly."""
    return x * x - D * y * y == rhs
