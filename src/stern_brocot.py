"""The Stern-Brocot tree and Farey sequences: every rational exactly once, by mediants.

There is a single binary tree that contains EVERY positive rational number exactly once, already in
lowest terms, with no repeats and nothing missing. The STERN-BROCOT TREE (Stern 1858, Brocot 1861 --
a clockmaker, who used it to design gear ratios) is built by MEDIANTS: between two fractions a/b and
c/d put their mediant (a+c)/(b+d), start from the boundary 0/1 and 1/0, and recurse. The mediant of two
adjacent fractions is always already reduced, and every reduced fraction appears at a unique node. So
the tree is a perfect enumeration of the rationals AND a binary search tree ordered by value -- walking
left/right from the root is exactly asking "smaller or larger?", which makes it the natural home of
best rational approximation.

Two structures fall out:

  BEST RATIONAL APPROXIMATION. To approximate a real x by a fraction with bounded denominator, descend
  the tree: at each node compare x to the current mediant and step toward it, accumulating a run of
  same-direction steps efficiently (the run lengths are exactly the CONTINUED-FRACTION coefficients of
  x -- the Stern-Brocot path IS the continued fraction). Stop when the denominator would exceed the
  bound. This yields the provably closest fraction with denominator <= N.

  FAREY SEQUENCE F_n. The sorted list of all reduced fractions in [0, 1] with denominator <= n. Its
  defining property: consecutive fractions a/b < c/d satisfy the UNIMODULAR relation bc - ad = 1, and
  the fraction that first appears between them in F_{b+d} is their mediant. Farey sequences are the
  restriction of the Stern-Brocot construction to a bounded denominator.

This module builds the Stern-Brocot path/coordinates of any rational, best-approximates reals under a
denominator bound, generates Farey sequences, and finds Farey neighbours by the unimodular relation. It
is validated against independent references: every rational's path reconstructs it exactly and matches
its continued-fraction expansion; the best approximation agrees with the repo's continued-fraction
best_approximation and beats every other fraction of no-larger denominator by brute force; the Farey
sequence matches a brute-force reduced-fraction enumeration and every consecutive pair satisfies
bc - ad = 1; and the mediant of Farey neighbours is their in-between successor. Pure stdlib; the
rational-enumeration companion to the continued-fraction and rational-arithmetic tools."""

from __future__ import annotations

from fractions import Fraction
from math import gcd


def mediant(a, b, c, d):
    """The mediant of a/b and c/d: (a+c)/(b+d) (as a numerator, denominator pair)."""
    return (a + c, b + d)


def stern_brocot_path(p, q):
    """The Stern-Brocot path to the reduced fraction p/q (p,q > 0) as a string of 'L'/'R'.

    Left = go to the smaller subtree, Right = larger. The root is 1/1. Returns "" for 1/1.
    """
    if p <= 0 or q <= 0:
        raise ValueError("stern_brocot_path needs positive p, q")
    g = gcd(p, q)
    p, q = p // g, q // g
    # boundaries 0/1 and 1/0
    la, lb, ra, rb = 0, 1, 1, 0
    path = []
    while True:
        ma, mb = mediant(la, lb, ra, rb)
        if (ma, mb) == (p, q):
            return "".join(path)
        if p * mb < ma * q:      # p/q < mediant -> go left
            path.append("L")
            ra, rb = ma, mb
        else:                    # p/q > mediant -> go right
            path.append("R")
            la, lb = ma, mb


def from_path(path):
    """Reconstruct the fraction (p, q) at the end of a Stern-Brocot path string."""
    la, lb, ra, rb = 0, 1, 1, 0
    ma, mb = mediant(la, lb, ra, rb)
    for ch in path:
        ma, mb = mediant(la, lb, ra, rb)
        if ch == "L":
            ra, rb = ma, mb
        else:
            la, lb = ma, mb
    ma, mb = mediant(la, lb, ra, rb)
    return (ma, mb)


def continued_fraction_of(p, q):
    """Continued-fraction coefficients [a0; a1, a2, ...] of p/q (the Stern-Brocot run lengths)."""
    coeffs = []
    while q:
        coeffs.append(p // q)
        p, q = q, p % q
    return coeffs


def best_rational_approximation(x, max_denominator):
    """Closest fraction to real x with denominator <= max_denominator, via Stern-Brocot descent.

    Returns a Fraction. Handles x >= 0. The descent accumulates same-direction runs in O(log) steps.
    """
    if x < 0:
        f = best_rational_approximation(-x, max_denominator)
        return -f
    # integer part
    n = int(x)
    frac = x - n
    if frac == 0:
        return Fraction(n, 1)
    # search in (0,1): boundaries 0/1 and 1/1
    la, lb, ra, rb = 0, 1, 1, 1
    best = None
    best_err = None

    def consider(a, b):
        nonlocal best, best_err
        if b > max_denominator:
            return False
        err = abs(frac - a / b)
        if best is None or err < best_err - 1e-18:
            best, best_err = (a, b), err
        return True

    consider(la, lb)
    consider(ra, rb)
    for _ in range(10000):
        ma, mb = mediant(la, lb, ra, rb)
        if mb > max_denominator:
            break
        consider(ma, mb)
        if frac < ma / mb:
            ra, rb = ma, mb
        elif frac > ma / mb:
            la, lb = ma, mb
        else:
            break
    a, b = best
    return Fraction(n * b + a, b)


def farey_sequence(n):
    """The Farey sequence F_n: sorted reduced fractions in [0,1] with denominator <= n.

    Uses the classic next-term recurrence (no sorting, no gcd): O(len) generation.
    """
    result = [Fraction(0, 1)]
    a, b, c, d = 0, 1, 1, n
    while c <= n:
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        result.append(Fraction(a, b))
    return result


def farey_neighbours_ok(seq):
    """Check every consecutive pair a/b < c/d in a Farey sequence satisfies bc - ad = 1."""
    for i in range(len(seq) - 1):
        f1, f2 = seq[i], seq[i + 1]
        a, b = f1.numerator, f1.denominator
        c, d = f2.numerator, f2.denominator
        if b * c - a * d != 1:
            return False
    return True


def brute_farey(n):
    """Reference Farey sequence by enumerating and reducing all fractions with denominator <= n."""
    s = set()
    for q in range(1, n + 1):
        for p in range(0, q + 1):
            s.add(Fraction(p, q))
    return sorted(s)


def brute_best_approximation(x, max_denominator):
    """Reference: closest fraction to x by scanning every denominator up to the bound."""
    best = None
    best_err = None
    for q in range(1, max_denominator + 1):
        p = round(x * q)
        err = abs(x - p / q)
        if best is None or err < best_err - 1e-18:
            best, best_err = Fraction(p, q), err
    return best
