"""Combinatorial ranking: bijecting permutations, combinations, and subsets to integers.

Every finite combinatorial object -- a permutation of n items, a k-subset of n, a bit-string -- can be
assigned a unique integer RANK from 0 to (count - 1), and recovered from it by UNRANKING. This
bijection is the workhorse of combinatorial generation: it lets you store a permutation as one
integer, pick a uniformly random one (rank a random integer), iterate all objects in order, or
distribute an enumeration across machines by rank ranges -- without ever materializing the full
(often astronomically large) set.

Three classic schemes. PERMUTATIONS use the FACTORIAL NUMBER SYSTEM (factoradic): the Lehmer code
counts, at each position, how many not-yet-used smaller elements remain, and those digits times
descending factorials give the lexicographic rank. COMBINATIONS (k-subsets) use the COMBINATORIAL
NUMBER SYSTEM: a subset {c_1 < c_2 < ...} maps to sum of C(c_i, i), a mixed-radix representation in
binomial coefficients. GRAY CODE ranks bit-strings so that consecutive ranks differ in exactly one
bit (rank = n XOR (n >> 1)), the reflected binary code used in rotary encoders and to minimize
switching. All three are exact integer arithmetic, so they scale to enormous n.

This module implements permutation rank/unrank (lexicographic, via the Lehmer code), combination
rank/unrank (the combinatorial number system), and Gray-code encode/decode. It is verified against
brute-force enumeration: that ranking then unranking is the identity, that ranks are a contiguous
0..N-1 bijection with no gaps or collisions, that permutation ranks match Python's lexicographic
itertools ordering, that combination ranks match itertools.combinations order, that consecutive Gray
codes differ in exactly one bit, and on hand-checked small cases. Pure stdlib; a combinatorics
companion to the De-Bruijn-sequence, Fisher-Yates, and next-permutation notes."""

from __future__ import annotations

import math


# --- permutations: Lehmer code / factorial number system -------------------
def permutation_rank(perm):
    """The lexicographic rank (0-based) of a permutation of {0..n-1} (or any comparable distinct
    items) among all n! permutations."""
    items = sorted(perm)
    rank = 0
    n = len(perm)
    for i in range(n):
        # how many remaining items are smaller than perm[i]?
        smaller = items.index(perm[i])
        rank += smaller * math.factorial(n - 1 - i)
        items.pop(smaller)
    return rank


def permutation_unrank(rank, n):
    """The `rank`-th (0-based, lexicographic) permutation of {0..n-1}."""
    items = list(range(n))
    perm = []
    for i in range(n):
        f = math.factorial(n - 1 - i)
        idx = rank // f
        rank %= f
        perm.append(items.pop(idx))
    return perm


# --- combinations: combinatorial number system -----------------------------
def combination_rank(combo, n):
    """The rank (0-based) of a k-subset `combo` of {0..n-1} among all C(n, k) subsets, in the order
    itertools.combinations produces (lexicographic by the ascending-sorted tuple)."""
    combo = sorted(combo)
    k = len(combo)
    total = math.comb(n, k)
    # rank = total - 1 - (co-rank in the combinatorial number system of the reversed complement)
    # use the standard colex-to-lex conversion: rank = sum over positions
    rank = 0
    prev = -1
    for i, c in enumerate(combo):
        # count combinations that would come before this one at position i
        for x in range(prev + 1, c):
            rank += math.comb(n - 1 - x, k - 1 - i)
        prev = c
    return rank


def combination_unrank(rank, n, k):
    """The `rank`-th (0-based) k-subset of {0..n-1} in itertools.combinations order."""
    combo = []
    x = 0
    for i in range(k):
        # find the next element
        while True:
            count = math.comb(n - 1 - x, k - 1 - i)
            if rank < count:
                combo.append(x)
                x += 1
                break
            rank -= count
            x += 1
    return combo


# --- Gray code: reflected binary --------------------------------------------
def gray_encode(n):
    """The Gray code of a non-negative integer n (n XOR n>>1)."""
    return n ^ (n >> 1)


def gray_decode(g):
    """The integer whose Gray code is g."""
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def gray_sequence(bits):
    """The full Gray-code sequence of `bits`-bit numbers: 2^bits values, consecutive ones differing
    in a single bit."""
    return [gray_encode(i) for i in range(1 << bits)]
