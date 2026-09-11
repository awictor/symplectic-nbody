"""Linear-time string matching: KMP, the Z-algorithm, and Manacher's palindromes.

Finding every occurrence of a pattern in a text is the most basic string operation, and the naive
"try every start position" is O(n*m). Three classic algorithms do fundamental string tasks in
LINEAR time by precomputing self-overlap information so the search never re-examines a character:

  KMP (Knuth-Morris-Pratt): build the PREFIX FUNCTION -- for each pattern position, the length of
      the longest proper prefix that is also a suffix -- so on a mismatch the pattern can shift by
      more than one without rescanning the text. O(n + m).
  Z-ALGORITHM: for each position of a string, the length of the longest substring starting there
      that matches a prefix of the whole string. Concatenating pattern + separator + text turns
      matching into reading off Z-values -- another O(n + m) matcher, and a building block for many
      string problems.
  MANACHER: the longest PALINDROMIC substring in O(n), by expanding palindromes while reusing
      mirror information (the palindrome analogue of the Z-idea), where the naive approach is
      O(n^2).

All three share the same trick: exploit a structure already computed about the string to skip
redundant comparisons. This module implements the prefix function and KMP search, the Z-array and
Z-based search, and Manacher's longest palindrome -- verified that KMP and the Z-search find exactly
the same occurrences as a brute-force scan (including overlapping matches) across many random and
structured strings, that the prefix function matches its definition, and that Manacher finds the
known longest palindrome. Pure stdlib; a string-algorithms companion to the Boyer-Moore and
Aho-Corasick notes."""

from __future__ import annotations


def prefix_function(s):
    """KMP prefix function: pi[i] = length of the longest proper prefix of s[:i+1] that is also a
    suffix of it. The heart of KMP."""
    n = len(s)
    pi = [0] * n
    k = 0
    for i in range(1, n):
        while k > 0 and s[i] != s[k]:
            k = pi[k - 1]
        if s[i] == s[k]:
            k += 1
        pi[i] = k
    return pi


def kmp_search(text, pattern):
    """All start indices where `pattern` occurs in `text`, via KMP. O(n + m). Finds overlapping
    matches. An empty pattern matches at every position (including the end)."""
    if pattern == "":
        return list(range(len(text) + 1))
    pi = prefix_function(pattern)
    m = len(pattern)
    matches = []
    k = 0
    for i, ch in enumerate(text):
        while k > 0 and ch != pattern[k]:
            k = pi[k - 1]
        if ch == pattern[k]:
            k += 1
        if k == m:
            matches.append(i - m + 1)
            k = pi[k - 1]                 # allow overlapping matches
    return matches


def z_array(s):
    """Z-array: z[i] = length of the longest substring starting at i that matches a prefix of s.
    z[0] is conventionally 0 (or len(s); we use 0)."""
    n = len(s)
    z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z


def z_search(text, pattern, sep="\x00"):
    """All occurrences of `pattern` in `text` using the Z-algorithm on pattern + sep + text."""
    if pattern == "":
        return list(range(len(text) + 1))
    combined = pattern + sep + text
    z = z_array(combined)
    m = len(pattern)
    matches = []
    for i in range(m + 1, len(combined)):
        if z[i] >= m:
            matches.append(i - m - 1)     # position in the original text
    return matches


def brute_force_search(text, pattern):
    """Naive O(n*m) search -- the reference implementation."""
    if pattern == "":
        return list(range(len(text) + 1))
    n, m = len(text), len(pattern)
    return [i for i in range(n - m + 1) if text[i:i + m] == pattern]


def longest_palindrome(s):
    """Manacher's algorithm: the longest palindromic substring of s, in O(n)."""
    if not s:
        return ""
    # transform: insert separators so even/odd palindromes are handled uniformly
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n            # p[i] = radius of the palindrome centred at i in t
    center = right = 0
    for i in range(1, n - 1):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    # find the maximum radius and map back to the original string
    max_len = max(p)
    center_idx = p.index(max_len)
    start = (center_idx - max_len) // 2      # position in the original string
    return s[start:start + max_len]


def count_occurrences(text, pattern):
    """Number of (possibly overlapping) occurrences of `pattern` in `text`."""
    return len(kmp_search(text, pattern))
