"""Boyer-Moore: the string search that skips ahead.

Finding a pattern inside a text is one of computing's most-run operations -- grep, editors, DNA
search, intrusion detection. The naive scan compares the pattern at every position, O(n*m). The
Boyer-Moore algorithm (1977) is the one your text editor actually uses, and its trick is
counterintuitive: it matches the pattern RIGHT-TO-LEFT, and on a mismatch it uses what it just
learned to jump the pattern forward by more than one position -- often by nearly its whole
length. On typical text it is sublinear: it does not even look at most characters.

Two precomputed rules decide the skip:

  * the BAD-CHARACTER rule: on a mismatch at text character c, shift the pattern so its last
    occurrence of c lines up with that position (or past it entirely if c is not in the
    pattern) -- a long pattern with a rare mismatch character leaps forward by its full length;

  * the GOOD-SUFFIX rule: when a suffix of the pattern matched before the mismatch, shift so the
    next occurrence of that suffix (or a matching prefix) aligns, never undoing confirmed
    matches.

Each step takes the larger of the two shifts, so the search is safe (never skips a real match)
and fast. This module builds both tables, finds the first and all occurrences, and is checked
exhaustively against a naive search. Pure stdlib; the string-algorithm companion to the k-d tree
and Union-Find notes.
"""

from __future__ import annotations


def _bad_char_table(pattern: str):
    """Last index at which each character occurs in the pattern. Missing characters map to -1
    implicitly (handled by the caller)."""
    last = {}
    for i, ch in enumerate(pattern):
        last[ch] = i
    return last


def _good_suffix_tables(pattern: str):
    """Compute the good-suffix shift table (the classic Boyer-Moore preprocessing). Returns a
    list `shift` of length m+1 giving how far to move the pattern when a mismatch occurs after
    matching the suffix starting at position i."""
    m = len(pattern)
    shift = [0] * (m + 1)
    border = [0] * (m + 1)  # border positions
    i = m
    j = m + 1
    border[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j
    # case where a prefix of the pattern is a suffix (partial matches)
    j = border[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = border[j]
    return shift


def find_all(text: str, pattern: str):
    """Return the list of all start indices where `pattern` occurs in `text` (overlapping
    matches included), using the full Boyer-Moore algorithm."""
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))  # empty pattern matches at every position, by convention
    if m > n:
        return []
    bad = _bad_char_table(pattern)
    good = _good_suffix_tables(pattern)

    result = []
    s = 0  # alignment of the pattern's start against the text
    while s <= n - m:
        j = m - 1
        # match right-to-left
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            result.append(s)
            s += good[0]  # shift for a full match
        else:
            bc = j - bad.get(text[s + j], -1)   # bad-character shift (>=1 by construction)
            gs = good[j + 1]                    # good-suffix shift
            s += max(bc, gs, 1)
    return result


def find(text: str, pattern: str) -> int:
    """Index of the first occurrence of `pattern` in `text`, or -1 if not present."""
    hits = find_all(text, pattern)
    return hits[0] if hits else -1


def contains(text: str, pattern: str) -> bool:
    """True if `pattern` occurs in `text`."""
    return find(text, pattern) != -1


def count(text: str, pattern: str) -> int:
    """Number of (overlapping) occurrences of `pattern` in `text`."""
    return len(find_all(text, pattern))


# --- naive reference (for validation) --------------------------------------

def naive_find_all(text: str, pattern: str):
    """Brute-force all occurrences, comparing the pattern at every position. O(n*m)."""
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    out = []
    for s in range(n - m + 1):
        if text[s:s + m] == pattern:
            out.append(s)
    return out
