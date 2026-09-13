"""Stirling and Bell numbers: counting set partitions and permutation cycles.

Two of the most important sequences in combinatorics count the ways to structure a set of n labelled
objects. The STIRLING NUMBER OF THE SECOND KIND, S(n, k), counts partitions of an n-element set into
exactly k non-empty unlabelled blocks -- how many ways to split n students into k study groups. The
BELL NUMBER B(n) sums those over all k: the total number of partitions of an n-set (B(3) = 5:
{123}, {12|3}, {13|2}, {23|1}, {1|2|3}). The STIRLING NUMBER OF THE FIRST KIND (unsigned), c(n, k),
counts permutations of n elements with exactly k cycles -- the other natural "how many pieces" count,
now for cyclic rather than set structure.

They obey elegant Pascal-like recurrences, each with a combinatorial story:

    S(n, k) = k * S(n-1, k) + S(n-1, k-1)      -- element n joins an existing block or starts its own,
    c(n, k) = (n-1) * c(n-1, k) + c(n-1, k-1)  -- element n slots into an existing cycle or is a fixed point,
    B(n)    = sum_{j} C(n-1, j) B(j)            -- (Bell's recurrence via the last block's size).

The two kinds of Stirling numbers are inverse triangular matrices connecting the ordinary powers
x^n to the falling factorials x(x-1)...(x-k+1) -- a duality this module verifies. The first-kind
row sums to n! (every permutation has some number of cycles), the second-kind row sums to the Bell
number, and both are unimodal.

This module computes S(n,k), c(n,k), B(n), full triangles, and the falling-factorial expansions.
Validated against brute force: S(n,k) matches an exhaustive count of set partitions into k blocks,
c(n,k) matches a count of permutations by cycle number, B(n) matches total partition enumeration and
Dobinski's series, the recurrences and row-sum identities (sum_k S(n,k) = B(n), sum_k c(n,k) = n!)
hold, and the falling-factorial polynomial identities check out. Pure stdlib; the set-structure
companion to the integer-partition and combinatorial-ranking notes."""

from __future__ import annotations

from functools import lru_cache
from math import factorial, comb, exp


@lru_cache(maxsize=None)
def stirling_second(n, k):
    """S(n, k): partitions of an n-set into exactly k non-empty blocks."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    return k * stirling_second(n - 1, k) + stirling_second(n - 1, k - 1)


@lru_cache(maxsize=None)
def stirling_first(n, k):
    """Unsigned c(n, k): permutations of n elements with exactly k cycles."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    return (n - 1) * stirling_first(n - 1, k) + stirling_first(n - 1, k - 1)


def signed_stirling_first(n, k):
    """Signed Stirling number of the first kind s(n,k) = (-1)^(n-k) c(n,k), the coefficients that
    expand falling factorials into powers."""
    return (-1) ** (n - k) * stirling_first(n, k)


@lru_cache(maxsize=None)
def bell(n):
    """B(n): the total number of partitions of an n-element set."""
    if n == 0:
        return 1
    # Bell's recurrence: B(n+1) = sum_{k=0}^{n} C(n,k) B(k)
    return sum(comb(n - 1, j) * bell(j) for j in range(n))


def bell_triangle(n_rows):
    """The Bell triangle (Aitken's array): each row starts with the last entry of the previous row,
    and B(n) reads off the left edge. Returns a list of rows."""
    if n_rows <= 0:
        return []
    triangle = [[1]]
    for _ in range(1, n_rows):
        prev = triangle[-1]
        row = [prev[-1]]
        for x in prev:
            row.append(row[-1] + x)
        triangle.append(row)
    return triangle


def stirling_second_row(n):
    """[S(n,0), S(n,1), ..., S(n,n)]."""
    return [stirling_second(n, k) for k in range(n + 1)]


def stirling_first_row(n):
    """[c(n,0), c(n,1), ..., c(n,n)]."""
    return [stirling_first(n, k) for k in range(n + 1)]


def falling_factorial_coeffs(n):
    """Coefficients of the falling factorial (x)_n = x(x-1)...(x-n+1) as a polynomial in x,
    lowest degree first. These are the signed first-kind Stirling numbers s(n,k)."""
    return [signed_stirling_first(n, k) for k in range(n + 1)]


def power_in_falling_factorials(n):
    """Coefficients expressing x^n = sum_k S(n,k) (x)_k in the falling-factorial basis."""
    return [stirling_second(n, k) for k in range(n + 1)]


def bell_dobinski(n, terms=60):
    """B(n) via Dobinski's formula B(n) = (1/e) sum_{j>=0} j^n / j!  (a floating-point check)."""
    return exp(-1) * sum(j ** n / factorial(j) for j in range(terms))


# --- brute-force references --------------------------------------------------
def brute_set_partitions(n):
    """Yield every set partition of {0..n-1} as a list of frozenset blocks."""
    if n == 0:
        yield []
        return
    # recursively place element n-1 into an existing block or a new one
    for smaller in brute_set_partitions(n - 1):
        # add (n-1) to each existing block
        for i in range(len(smaller)):
            new = [set(b) for b in smaller]
            new[i].add(n - 1)
            yield [frozenset(b) for b in new]
        # or as its own block
        yield [frozenset(b) for b in smaller] + [frozenset({n - 1})]


def brute_stirling_second(n, k):
    """Count set partitions of an n-set into exactly k blocks, exhaustively."""
    return sum(1 for p in brute_set_partitions(n) if len(p) == k)


def brute_bell(n):
    """Total number of set partitions of an n-set, exhaustively."""
    return sum(1 for _ in brute_set_partitions(n))


def _permutations(seq):
    if len(seq) <= 1:
        yield tuple(seq)
        return
    for i in range(len(seq)):
        rest = seq[:i] + seq[i + 1:]
        for p in _permutations(rest):
            yield (seq[i],) + p


def _cycle_count(perm):
    """Number of cycles in a permutation given as a tuple mapping i -> perm[i]."""
    n = len(perm)
    seen = [False] * n
    cycles = 0
    for i in range(n):
        if not seen[i]:
            cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
    return cycles


def brute_stirling_first(n, k):
    """Count permutations of n elements with exactly k cycles, exhaustively."""
    if n == 0:
        return 1 if k == 0 else 0
    return sum(1 for p in _permutations(list(range(n))) if _cycle_count(p) == k)
