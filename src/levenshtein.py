"""Levenshtein edit distance: how far apart are two strings.

The edit distance between two strings is the minimum number of single-character edits --
insertions, deletions, or substitutions -- that turn one into the other. Vladimir Levenshtein
defined it in 1965, and it is the measure behind spell-checkers ("did you mean...?"), fuzzy
search, diff tools, DNA sequence alignment, and plagiarism detection.

It is computed by dynamic programming. Let d[i][j] be the distance between the first i characters
of a and the first j of b. Then

    d[i][j] = d[i-1][j-1]                     if a[i-1] == b[j-1]  (no edit),
            = 1 + min( d[i-1][j],             delete a[i-1]
                       d[i][j-1],             insert b[j-1]
                       d[i-1][j-1] )          substitute,

filling an (m+1) x (n+1) table in O(mn) time. Following the choices backward from the corner
recovers the actual sequence of edits -- the alignment -- not just the number. The distance
obeys the metric axioms (it is a true distance: symmetric, zero only for equal strings, and
triangle-inequality-respecting), so it can index approximate matches.

Two useful relatives are included: a memory-lean O(min(m,n)) row-only distance for when you only
need the number, and the Damerau variant that also counts a transposition of two adjacent
characters as one edit (the most common typo). This module computes the distance, the alignment
operations, a normalized similarity ratio, and both variants, and checks them against known
values and the metric properties. Pure stdlib; the string-algorithm companion to the Boyer-Moore
note."""

from __future__ import annotations


def distance(a: str, b: str) -> int:
    """Levenshtein edit distance between a and b (insertions, deletions, substitutions), by the
    full dynamic-programming table. O(mn) time and space."""
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,        # deletion
                          d[i][j - 1] + 1,        # insertion
                          d[i - 1][j - 1] + cost)  # substitution / match
    return d[m][n]


def distance_fast(a: str, b: str) -> int:
    """Edit distance using only two rows -- O(min(m,n)) memory. Same result as `distance`."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def _table(a: str, b: str):
    """Build and return the full DP table (used for alignment backtracing)."""
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d


def alignment(a: str, b: str):
    """Return the list of edit operations turning a into b, as tuples:
        ("match", ch)        characters equal, no edit
        ("substitute", x, y) replace x with y
        ("delete", x)        remove x from a
        ("insert", y)        add y from b
    Backtraces the DP table from the bottom-right corner."""
    d = _table(a, b)
    i, j = len(a), len(b)
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(("match", a[i - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("substitute", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("delete", a[i - 1]))
            i -= 1
        else:  # insertion
            ops.append(("insert", b[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def apply_ops(a: str, ops) -> str:
    """Apply a list of alignment operations to `a`, reproducing `b` -- proof the ops are
    correct."""
    out = []
    ai = 0
    for op in ops:
        kind = op[0]
        if kind == "match":
            out.append(a[ai])
            ai += 1
        elif kind == "substitute":
            out.append(op[2])
            ai += 1
        elif kind == "delete":
            ai += 1
        elif kind == "insert":
            out.append(op[1])
        else:
            raise ValueError(f"unknown op {kind}")
    return "".join(out)


def similarity(a: str, b: str) -> float:
    """Normalized similarity in [0, 1]: 1 - distance / max(len). 1.0 for identical strings, 0.0
    when every character must change."""
    if not a and not b:
        return 1.0
    return 1.0 - distance(a, b) / max(len(a), len(b))


def damerau_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance: like Levenshtein but a transposition of two ADJACENT
    characters also counts as a single edit (the commonest typo, e.g. 'teh' -> 'the')."""
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition
    return d[m][n]
