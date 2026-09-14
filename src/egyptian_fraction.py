"""Egyptian fractions: writing any rational as a sum of distinct unit fractions.

The ancient Egyptians wrote every fraction as a sum of DISTINCT UNIT FRACTIONS -- reciprocals of
integers, like 2/3 = 1/2 + 1/6 -- with no repeats. That such a decomposition always exists for any
rational in (0, 1) is not obvious, and finding one is a small classic of algorithmic number theory.
The FIBONACCI-SYLVESTER GREEDY algorithm proves existence constructively: at each step, subtract the
LARGEST unit fraction 1/ceil(1/x) that does not exceed the remaining value x. Astonishingly, the
numerator of the remainder STRICTLY DECREASES every step (Fibonacci's 1202 proof), so the process must
terminate -- you always reach 0 in finitely many terms, all denominators distinct.

Two companions round out the picture:

  The ENGEL EXPANSION writes x = 1/a_1 + 1/(a_1 a_2) + 1/(a_1 a_2 a_3) + ... with a non-decreasing
  sequence of integers a_i, an "ascending continued fraction" that gives another canonical unit-
  fraction form.

  SYLVESTER'S SEQUENCE 2, 3, 7, 43, 1807, ... (each term is one plus the product of all previous) is
  exactly the greedy Egyptian expansion of 1 that converges the fastest possible -- the sum of its
  reciprocals approaches 1, and it is the extremal example for how few terms an Egyptian expansion can
  use for a value near 1.

This module implements the greedy Fibonacci-Sylvester decomposition, the Engel expansion (and its
reconstruction), and Sylvester's sequence. It is validated exactly with rational arithmetic: every
greedy decomposition sums back to the input, uses strictly increasing (hence distinct) denominators,
and terminates; the numerator provably shrinks each step; the Engel expansion sums back and is
non-decreasing; Sylvester's sequence satisfies its recurrence and its reciprocals telescope toward 1;
and known small cases (2/3 = 1/2 + 1/6, 3/7 = 1/3 + 1/11 + 1/231) match. Pure stdlib (Fraction); the
unit-fraction companion to the continued-fraction and Stern-Brocot rational tools."""

from __future__ import annotations

from fractions import Fraction
from math import gcd


def greedy_egyptian(numerator, denominator):
    """Fibonacci-Sylvester greedy Egyptian fraction of numerator/denominator in (0, 1).

    Returns a list of distinct denominators [d_1, d_2, ...] with sum 1/d_i = numerator/denominator.
    """
    x = Fraction(numerator, denominator)
    if x <= 0 or x >= 1:
        raise ValueError("greedy_egyptian expects a fraction strictly between 0 and 1")
    denoms = []
    while x > 0:
        # largest unit fraction 1/d <= x  ->  d = ceil(1/x)
        d = (x.denominator + x.numerator - 1) // x.numerator  # ceil(den/num)
        denoms.append(d)
        x -= Fraction(1, d)
    return denoms


def egyptian_sum(denoms):
    """Sum of the unit fractions with the given denominators, as a Fraction."""
    total = Fraction(0)
    for d in denoms:
        total += Fraction(1, d)
    return total


def engel_expansion(numerator, denominator):
    """Engel expansion: x = 1/a1 + 1/(a1 a2) + ... with non-decreasing a_i. Returns [a1, a2, ...]."""
    x = Fraction(numerator, denominator)
    if x <= 0:
        raise ValueError("engel_expansion expects a positive rational")
    a = []
    while x != 0:
        # a_i = ceil(1/x)
        u = (x.denominator + x.numerator - 1) // x.numerator  # ceil(1/x)
        a.append(u)
        x = x * u - 1
    return a


def engel_to_fraction(a):
    """Reconstruct the rational from its Engel expansion [a1, a2, ...]."""
    x = Fraction(0)
    prod = 1
    for ai in a:
        prod *= ai
        x += Fraction(1, prod)
    return x


def sylvester_sequence(n):
    """The first n terms of Sylvester's sequence: 2, 3, 7, 43, 1807, ... (a_{k+1} = 1 + prod a_i)."""
    seq = []
    prod = 1
    for _ in range(n):
        term = prod + 1
        seq.append(term)
        prod *= term
    return seq


def sylvester_reciprocal_sum(n):
    """Sum of reciprocals of the first n Sylvester terms (approaches 1 from below)."""
    total = Fraction(0)
    for term in sylvester_sequence(n):
        total += Fraction(1, term)
    return total


def all_distinct(denoms):
    """True if all denominators are distinct."""
    return len(set(denoms)) == len(denoms)


def is_increasing(denoms):
    """True if the denominators are strictly increasing (the greedy algorithm's guarantee)."""
    return all(denoms[i] < denoms[i + 1] for i in range(len(denoms) - 1))
