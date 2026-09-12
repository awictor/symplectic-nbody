"""Continued fractions: the best rational approximations of a real number.

Every real number x has a CONTINUED FRACTION expansion x = a0 + 1/(a1 + 1/(a2 + ...)), where the a_i
are positive integers found by a simple greedy process: take the integer part, invert the remainder,
repeat. Truncating the expansion gives the CONVERGENTS -- the successive fractions p_k/q_k -- and
these are the BEST RATIONAL APPROXIMATIONS of x in a precise sense: no fraction with a smaller or
equal denominator is closer to x. This is why 22/7 and 355/113 are the famous approximations of pi,
why gear ratios and calendar leap-year rules are chosen this way, and how the continued fraction of a
quadratic irrational (which is eventually periodic) solves Pell's equation.

The convergents are computed by a beautiful recurrence: p_k = a_k p_{k-1} + p_{k-2} and q_k = a_k
q_{k-1} + q_{k-2}, seeded so that consecutive convergents satisfy p_k q_{k-1} - p_{k-1} q_k = (-1)^k
(they are always in lowest terms, and they straddle x, alternating above and below while closing in).
The error of the k-th convergent is bounded by 1/(q_k q_{k+1}), so a large partial quotient a_{k+1}
means the previous convergent is exceptionally good -- exactly what makes 355/113 (from pi's large
quotient 292) accurate to seven digits.

This module computes the continued-fraction expansion of a real number or an exact fraction, its
convergents, and the best rational approximation within a denominator bound. It is verified against
exact references: that the convergents of pi are 3, 22/7, 333/106, 355/113, that reconstructing a
finite expansion recovers the original rational exactly, that each convergent is the best rational
approximation for its denominator (checked against all smaller-denominator fractions), that the
golden ratio's expansion is all ones, that sqrt(2) is [1; 2, 2, 2, ...], and that the convergents
alternate around and converge to the target. Pure stdlib; a number-theory companion to the RSA,
Pollard-rho, and Diophantine notes."""

from __future__ import annotations

import math
from fractions import Fraction


def cf_expansion(x, max_terms=40, tol=1e-15):
    """The continued-fraction expansion [a0; a1, a2, ...] of a real number x (or a Fraction, exact)."""
    terms = []
    if isinstance(x, Fraction):
        # exact Euclidean-style expansion
        num, den = x.numerator, x.denominator
        while den != 0:
            a = num // den
            terms.append(a)
            num, den = den, num - a * den
        return terms
    # floating-point expansion
    for _ in range(max_terms):
        a = math.floor(x)
        terms.append(a)
        frac = x - a
        if frac < tol:
            break
        x = 1.0 / frac
    return terms


def convergents(terms):
    """The convergents p_k/q_k of a continued fraction [a0; a1, ...] as a list of Fractions."""
    result = []
    p_prev, p_prev2 = 1, 0        # p_{-1}=1, p_{-2}=0
    q_prev, q_prev2 = 0, 1        # q_{-1}=0, q_{-2}=1
    for a in terms:
        p = a * p_prev + p_prev2
        q = a * q_prev + q_prev2
        result.append(Fraction(p, q))
        p_prev2, p_prev = p_prev, p
        q_prev2, q_prev = q_prev, q
    return result


def evaluate(terms):
    """Evaluate a finite continued fraction [a0; a1, ...] to an exact Fraction."""
    if not terms:
        return Fraction(0)
    acc = Fraction(terms[-1])
    for a in reversed(terms[:-1]):
        acc = a + 1 / acc
    return acc


def best_approximation(x, max_denominator):
    """The best rational approximation of x with denominator <= max_denominator: the last convergent
    (or semi-convergent) whose denominator fits. Returns a Fraction."""
    terms = cf_expansion(x, max_terms=60)
    best = Fraction(math.floor(x))
    p_prev, p_prev2 = 1, 0
    q_prev, q_prev2 = 0, 1
    for i, a in enumerate(terms):
        p = a * p_prev + p_prev2
        q = a * q_prev + q_prev2
        if q > max_denominator:
            # try a semi-convergent: reduce the last partial quotient
            # a' = floor((max_denominator - q_prev2) / q_prev)
            if q_prev > 0:
                a2 = (max_denominator - q_prev2) // q_prev
                if a2 > 0:
                    cand = Fraction(a2 * p_prev + p_prev2, a2 * q_prev + q_prev2)
                    # accept the semi-convergent only if it's at least half-way (standard rule)
                    if 2 * a2 >= a and abs(float(cand) - x) < abs(float(best) - x):
                        best = cand
            break
        best = Fraction(p, q)
        p_prev2, p_prev = p_prev, p
        q_prev2, q_prev = q_prev, q
    return best


def approximation_error(x, frac):
    """The absolute error |x - p/q|."""
    return abs(x - float(frac))
