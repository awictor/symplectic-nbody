"""Longest common subsequence: the shared thread through two sequences.

A subsequence keeps some elements in order but may skip others (unlike a substring, it need not
be contiguous). The longest common subsequence (LCS) of two sequences is the longest ordering of
elements appearing, in that order, in both -- and it is the engine behind `diff`, version-control
merges, bioinformatics sequence comparison, and the Unix `patch` tool. The complement of the LCS
is exactly the set of lines you have to add or delete to turn one file into the other, so a
bigger LCS means a smaller diff.

It is a dynamic program. Let L[i][j] be the LCS length of the first i elements of a and the first
j of b:

    L[i][j] = L[i-1][j-1] + 1                 if a[i-1] == b[j-1]  (extend the match),
            = max( L[i-1][j], L[i][j-1] )     otherwise            (drop one element),

filling an (m+1) x (n+1) table in O(mn). Backtracing the choices from the corner recovers an
actual longest subsequence, and turning the same walk into "keep / delete from a / insert from b"
steps yields a diff. The LCS length also gives the edit distance under insert/delete-only edits:
`m + n - 2*LCS`.

This module computes the LCS length, one longest subsequence, the diff edit-script, and the
insert/delete edit distance, and checks them against a brute-force search over all subsequences.
Pure stdlib; the dynamic-programming companion to the Levenshtein and knapsack notes.
"""

from __future__ import annotations


def _table(a, b):
    """Build the LCS-length DP table for sequences a and b."""
    m, n = len(a), len(b)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = L[i - 1][j] if L[i - 1][j] >= L[i][j - 1] else L[i][j - 1]
    return L


def lcs_length(a, b) -> int:
    """Length of the longest common subsequence of a and b. O(mn)."""
    return _table(a, b)[len(a)][len(b)]


def lcs(a, b):
    """One longest common subsequence of a and b, as a list of elements (a string if both inputs
    are strings). Backtraces the DP table."""
    L = _table(a, b)
    i, j = len(a), len(b)
    out = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i, j = i - 1, j - 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    out.reverse()
    if isinstance(a, str) and isinstance(b, str):
        return "".join(out)
    return out


def diff(a, b):
    """A diff edit-script turning a into b, as a list of operations:
        ("keep", x)     x is in both (part of the LCS),
        ("delete", x)   x removed from a,
        ("insert", y)   y added from b.
    Built by backtracing the LCS table -- the same alignment `diff` and `patch` use."""
    L = _table(a, b)
    i, j = len(a), len(b)
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append(("keep", a[i - 1]))
            i, j = i - 1, j - 1
        elif j > 0 and (i == 0 or L[i][j - 1] >= L[i - 1][j]):
            ops.append(("insert", b[j - 1]))
            j -= 1
        else:
            ops.append(("delete", a[i - 1]))
            i -= 1
    ops.reverse()
    return ops


def apply_diff(a, ops):
    """Apply a diff edit-script to `a`, reproducing `b` -- proof the script is correct."""
    out = []
    ai = 0
    for kind, val in ops:
        if kind == "keep":
            out.append(a[ai]); ai += 1
        elif kind == "delete":
            ai += 1
        elif kind == "insert":
            out.append(val)
        else:
            raise ValueError(f"unknown op {kind}")
    if isinstance(a, str):
        return "".join(out)
    return out


def edit_distance_indel(a, b) -> int:
    """Insert/delete-only edit distance (no substitutions): m + n - 2*LCS. The number of lines
    a diff must add or remove."""
    return len(a) + len(b) - 2 * lcs_length(a, b)


# --- brute-force reference (for validation) --------------------------------

def brute_lcs_length(a, b) -> int:
    """LCS length by checking every subsequence of the shorter sequence -- exponential, for
    tests only."""
    if len(a) > len(b):
        a, b = b, a
    n = len(a)
    best = 0
    for mask in range(1 << n):
        sub = [a[i] for i in range(n) if mask & (1 << i)]
        if _is_subsequence(sub, b):
            best = max(best, len(sub))
    return best


def _is_subsequence(sub, seq) -> bool:
    """True if `sub` appears in `seq` in order (not necessarily contiguously)."""
    it = iter(seq)
    return all(x in it for x in sub)
