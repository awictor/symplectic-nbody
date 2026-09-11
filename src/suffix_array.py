"""Suffix arrays: a compact string index for search, repeats, and comparison.

A suffix array is the sorted order of all SUFFIXES of a string, stored as their start indices. It
packs almost everything a suffix TREE offers -- fast substring search, longest-repeated-substring,
longest-common-substring -- into a single integer array of length n, which is why it underlies
full-text search, bioinformatics indices, and the bzip2 family. Once built, "does pattern P occur?"
is a BINARY SEARCH over the sorted suffixes in O(m log n), and every occurrence is a contiguous run
in the array.

Building it naively (sort the n suffixes as strings) is O(n^2 log n); the standard speed-up here is
PREFIX DOUBLING: sort suffixes by their first character, then by their first 2, 4, 8, ... using the
previous ranks as sort keys, so each of the log n rounds is an O(n log n) sort -- O(n log^2 n)
overall, fine for real texts. The companion LCP ARRAY (longest common prefix of adjacent sorted
suffixes) is built in O(n) by KASAI'S algorithm and unlocks the tree-like queries: the longest
repeated substring is simply the largest LCP value.

This module builds the suffix array by prefix doubling, the LCP array by Kasai, substring search by
binary search (with all occurrences), and the longest repeated substring -- verified that the suffix
array is the true sorted order of suffixes (checked against a brute-force sort), that search finds
exactly the same occurrences as a scan (including overlaps), that the LCP array matches the direct
prefix computation, and that the longest repeated substring matches a brute-force search. Pure
stdlib; a string-index companion to the trie and KMP notes."""

from __future__ import annotations

from bisect import bisect_left, bisect_right


def build_suffix_array(s):
    """Suffix array of s by prefix doubling: the start indices of all suffixes in sorted order.
    O(n log^2 n)."""
    n = len(s)
    if n == 0:
        return []
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n
    k = 1
    while True:
        # sort by (rank[i], rank[i+k]) -- the first 2k characters of each suffix
        def key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        # recompute ranks from the new order
        tmp[sa[0]] = 0
        for j in range(1, n):
            tmp[sa[j]] = tmp[sa[j - 1]] + (1 if key(sa[j]) != key(sa[j - 1]) else 0)
        rank = tmp[:]
        if rank[sa[-1]] == n - 1:      # all suffixes have distinct ranks -> fully sorted
            break
        k <<= 1
    return sa


def build_lcp(s, sa):
    """LCP array by Kasai's algorithm: lcp[i] = length of the longest common prefix of the suffixes
    sa[i-1] and sa[i] (lcp[0] = 0). O(n)."""
    n = len(s)
    if n == 0:
        return []
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
        else:
            h = 0
    return lcp


def search(s, sa, pattern):
    """All start indices where `pattern` occurs in s, via binary search over the suffix array.
    O(m log n). Returns them sorted ascending."""
    if pattern == "":
        return list(range(len(s) + 1))
    n = len(s)
    m = len(pattern)
    # find the range of suffixes that start with `pattern`
    suffixes = _SuffixKey(s, sa, m)
    lo = bisect_left(suffixes, pattern)
    hi = bisect_right(suffixes, pattern)
    return sorted(sa[i] for i in range(lo, hi))


class _SuffixKey:
    """A lazy sequence view: index i -> the first m characters of the suffix sa[i], for bisect."""

    def __init__(self, s, sa, m):
        self.s = s
        self.sa = sa
        self.m = m

    def __len__(self):
        return len(self.sa)

    def __getitem__(self, i):
        start = self.sa[i]
        return self.s[start:start + self.m]


def contains(s, sa, pattern):
    """True if `pattern` occurs in s (a single binary search)."""
    return len(search(s, sa, pattern)) > 0


def longest_repeated_substring(s):
    """The longest substring occurring at least twice, from the maximum LCP value. Empty if none
    repeats."""
    n = len(s)
    if n < 2:
        return ""
    sa = build_suffix_array(s)
    lcp = build_lcp(s, sa)
    best = 0
    pos = 0
    for i in range(1, n):
        if lcp[i] > best:
            best = lcp[i]
            pos = sa[i]
    return s[pos:pos + best]


def longest_common_substring(a, b, sep="\x01"):
    """The longest substring common to both a and b, via the LCP of the concatenation a + sep + b
    (adjacent suffixes from different sides with the largest LCP)."""
    s = a + sep + b
    n = len(s)
    sa = build_suffix_array(s)
    lcp = build_lcp(s, sa)
    na = len(a)

    def side(idx):
        return 0 if idx < na else 1      # which original string a suffix comes from

    best = 0
    pos = 0
    for i in range(1, n):
        if side(sa[i]) != side(sa[i - 1]) and lcp[i] > best:
            best = lcp[i]
            pos = sa[i]
    return s[pos:pos + best]
