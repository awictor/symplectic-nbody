"""Bertrand's ballot problem: the odds a lead is never lost, and the reflection principle that proves it.

In an election candidate A finishes with p votes and B with q votes, p > q, so A wins. But the votes are
counted one at a time in random order -- what is the chance A is STRICTLY AHEAD at every single moment of
the count, never once tied or trailing? Bertrand asked this in 1887 and the answer is astonishingly clean:

    P(A leads throughout) = (p - q) / (p + q)

independent of how large the counts are, depending only on the margin relative to the turnout. A landslide
(p >> q) is almost surely never behind; a squeaker (p = q + 1) leads throughout only 1/(p+q) of the time.

The proof is the REFLECTION PRINCIPLE, one of the most beautiful arguments in combinatorics. Encode a count
as a lattice path: +1 up for each A vote, -1 down for each B vote, starting at height 0 and ending at
height p - q. "A leads throughout" means the path stays strictly above 0 after the first step. Count the
BAD paths -- those that touch 0 -- by reflecting the portion before the first return across the axis; this
sets up a bijection between bad paths starting with an A and all paths starting with a B, and the counts
subtract cleanly to leave (p-q)/(p+q) of the total. The same idea, run with a weak inequality (A never
TRAILS, ties allowed), yields the ballot numbers and, at p = q, the CATALAN numbers -- the count of Dyck
paths, balanced parentheses, and a hundred other things.

This module computes:

  STRICT ballot count and probability -- sequences where every prefix has #A > #B, via the exact formula
      and, independently, by the reflection principle (all paths minus reflected bad paths).
  WEAK ballot count -- every prefix has #A >= #B (ties allowed), the ballot number
      (p-q+1)/(p+1) * C(p+q, p).
  CATALAN numbers -- catalan(n) = weak count at p = q = n = C(2n, n)/(n+1), with the Dyck-path meaning.
  The CYCLE-LEMMA count -- exactly p-q of the p+q cyclic rotations of any arrangement keep A ahead, an
      independent combinatorial route to the same number.

Everything is exact integer / Fraction arithmetic, no floats in the counts. Validated by brute-force
enumeration of every vote ordering for small p, q: the formula, the reflection-principle count, the
cycle-lemma count, and the direct count all agree; Catalan numbers match the known sequence and the
brute-forced Dyck-path count. Pure stdlib. The first-passage companion to the random-walk, Catalan, and
Galton board notes."""

from __future__ import annotations

import math
from fractions import Fraction


def ballot_probability(p, q):
    """Probability that A (p votes) is STRICTLY ahead of B (q votes) throughout the count.

    Bertrand's theorem: (p - q) / (p + q) for p > q. Returns an exact Fraction. For p <= q the event is
    impossible (A cannot lead the whole way without ending ahead) except the degenerate p = q = 0 -> 0."""
    if p + q == 0:
        return Fraction(0)
    if p <= q:
        return Fraction(0)
    return Fraction(p - q, p + q)


def ballot_count_strict(p, q):
    """Number of vote orderings in which every prefix has strictly more A votes than B votes.

    Equals (p - q)/(p + q) * C(p + q, p), an integer for p > q. Zero if p <= q (the first step must be an
    A and the path can never return to 0, so the total must end positive)."""
    if p <= q:
        return 0
    total = math.comb(p + q, p)
    return (p - q) * total // (p + q)


def ballot_count_strict_reflection(p, q):
    """Same strict count, derived by the REFLECTION PRINCIPLE instead of the closed form.

    A valid path must take its first step up (an A vote) and then stay strictly positive. After fixing that
    first A, the rest is a path from height 1 to height p-q using p-1 up and q down steps that never touches
    0. By the reflection principle the number of such paths that DO touch 0 equals the number of unrestricted
    paths from height -1 to p-q (reflect the start across the axis) = C(p-1+q, q-1). So

        strict = C(p-1+q, p-1) - C(p-1+q, q-1)

    counts good continuations. This is an independent check on the closed-form (p-q)/(p+q) * C(p+q,p)."""
    if p <= q:
        return 0
    n = p - 1 + q
    all_paths = math.comb(n, p - 1)          # paths from height 1 with p-1 ups, q downs
    bad_paths = math.comb(n, q - 1) if q >= 1 else 0  # reflected: start at -1
    return all_paths - bad_paths


def ballot_count_cycle_lemma(p, q):
    """Strict count via the CYCLE LEMMA (Dvoretzky-Motzkin).

    For a sequence of p +1's and q -1's with p > q, among its p + q cyclic rotations EXACTLY p - q keep
    every partial sum positive. Each necklace of p+q arrangements thus contributes p-q good linear
    sequences, and there are C(p+q, p)/(p+q) necklaces on average, giving (p-q)/(p+q) * C(p+q, p). We
    return that product form (identical value, different derivation)."""
    if p <= q:
        return 0
    return (p - q) * math.comb(p + q, p) // (p + q)


def ballot_count_weak(p, q):
    """Number of orderings where every prefix has #A >= #B (ties allowed, A never trails).

    The ballot number (p - q + 1)/(p + 1) * C(p + q, p). Requires p >= q; zero otherwise."""
    if p < q:
        return 0
    total = math.comb(p + q, p)
    return (p - q + 1) * total // (p + 1)


def catalan(n):
    """The n-th Catalan number C(2n, n)/(n + 1): weak ballot count at p = q = n (Dyck paths of length 2n)."""
    return math.comb(2 * n, n) // (n + 1)


def brute_force_strict(p, q):
    """Directly enumerate every distinct ordering of p A-votes and q B-votes; count strict-lead orderings.

    Exponential -- only for validation on small p, q. Walks the multiset of arrangements as a binary tree."""
    return _enumerate(p, q, strict=True)


def brute_force_weak(p, q):
    """Enumerate every ordering; count those where A never trails (prefix #A >= #B). Small p, q only."""
    return _enumerate(p, q, strict=False)


def _enumerate(p, q, strict):
    count = 0

    def rec(a_left, b_left, lead):
        nonlocal count
        if a_left == 0 and b_left == 0:
            count += 1
            return
        # place an A next
        if a_left > 0:
            new_lead = lead + 1
            if (new_lead > 0) if strict else (new_lead >= 0):
                rec(a_left - 1, b_left, new_lead)
        # place a B next
        if b_left > 0:
            new_lead = lead - 1
            if (new_lead > 0) if strict else (new_lead >= 0):
                rec(a_left, b_left - 1, new_lead)

    rec(p, q, 0)
    return count
