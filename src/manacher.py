"""Manacher's algorithm: every palindromic substring in linear time.

Finding the LONGEST PALINDROMIC SUBSTRING of a string -- the longest stretch that reads the same
forwards and backwards -- is a classic problem with a naive O(n^2) or O(n^3) solution: try every
center or every substring and check. MANACHER'S ALGORITHM (1975) does it in O(n) by computing, for
every position, the RADIUS of the longest palindrome centered there, reusing the symmetry of
already-discovered palindromes so no character is examined more than a constant number of times. It
is the definitive linear-time palindrome algorithm, used in bioinformatics (finding reverse-
complement structure), text processing, and competitive programming.

The trick handles even- and odd-length palindromes uniformly by TRANSFORMING the string: insert a
separator (say '|') between every character and at the ends, so "aba" becomes "|a|b|a|"; now every
palindrome has an odd length in the transformed string and a well-defined single center. The core is
a running scan that maintains the RIGHTMOST palindrome found so far (its center C and right edge R).
For a new center i inside R, its palindrome radius is at least that of its MIRROR position 2C - i
(clamped to R), so the algorithm starts from that free lower bound and only expands beyond it when
possible -- and each expansion advances R, so the total expansion work is O(n). Reading the radii
back gives the longest palindrome, the count of all palindromic substrings, and a test for any
substring.

This module implements Manacher's algorithm returning the per-center radii, the longest palindromic
substring, the total number of palindromic substrings, and all maximal palindromes. It is verified
against brute force: that the longest palindrome matches an O(n^2) center-expansion reference, that
the total palindrome count matches a brute enumeration over all substrings, that known strings give
the expected answers, and on edge cases (empty, single character, all-same, no palindromes longer
than one). Pure stdlib; a string-algorithm companion to the suffix-array, KMP, and Aho-Corasick
notes."""

from __future__ import annotations


def _radii(s):
    """Manacher radii on the separator-transformed string. Returns (transformed, radii) where
    radii[i] is the palindrome radius centered at position i of the transformed string."""
    t = "|" + "|".join(s) + "|"
    n = len(t)
    p = [0] * n
    center = right = 0
    for i in range(n):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        # expand around i
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    return t, p


def longest_palindrome(s):
    """The longest palindromic substring of s (the first one if there are ties)."""
    if not s:
        return ""
    t, p = _radii(s)
    max_len = max(p)
    center_idx = p.index(max_len)
    # map back to the original string: the palindrome spans [ (center-radius)/2 , ... )
    start = (center_idx - max_len) // 2
    return s[start:start + max_len]


def count_palindromic_substrings(s):
    """The total number of palindromic substrings (each distinct start,end position counts once)."""
    if not s:
        return 0
    _, p = _radii(s)
    # each center contributes ceil(radius / 2) palindromic substrings in the original string
    return sum((r + 1) // 2 for r in p)


def all_palindrome_radii(s):
    """For each center in the original string, the radius of the longest palindrome there.

    Returns two lists: odd_radii[i] (odd-length palindromes centered at char i, radius in chars) and
    even_radii[i] (even-length palindromes centered between chars i and i+1)."""
    if not s:
        return [], []
    t, p = _radii(s)
    n = len(s)
    odd = [0] * n
    even = [0] * (n - 1) if n > 1 else []
    for i in range(len(t)):
        # centers at odd positions of t are original characters; even positions are between chars
        if i % 2 == 1:                      # a character center -> odd-length palindrome
            char_idx = (i - 1) // 2
            odd[char_idx] = (p[i] + 1) // 2
        else:                               # a gap center -> even-length palindrome
            gap_idx = i // 2 - 1
            if 0 <= gap_idx < len(even):
                even[gap_idx] = p[i] // 2
    return odd, even


def is_palindrome(s):
    """True if the whole string is a palindrome."""
    return s == s[::-1]
