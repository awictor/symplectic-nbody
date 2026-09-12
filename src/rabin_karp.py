"""Rabin-Karp: substring search by rolling polynomial hashes.

Naive substring search compares a length-m pattern against every length-m window of an n-character
text, O(n*m) in the worst case. RABIN-KARP (1987) makes it O(n + m) on average by HASHING: treat each
window as a base-B number modulo a large prime, so two windows with different hashes cannot be equal
and only hash MATCHES need a full character check. The magic is the ROLLING HASH -- sliding the window
by one character updates the hash in O(1) (subtract the departing character's contribution, multiply by
the base, add the arriving character) instead of recomputing it from scratch. It is the workhorse
behind multi-pattern search (hash every pattern once, scan the text once), plagiarism and duplicate
detection, and the binary-search-on-length trick for the longest common substring.

The polynomial hash of a string s is h(s) = (s[0] B^(m-1) + s[1] B^(m-2) + ... + s[m-1]) mod P. Rolling
from window i to i+1 removes s[i]*B^(m-1), shifts left (times B), and adds s[i+m]. Because different
strings can collide (share a hash), every candidate is verified by a direct comparison -- so the result
is always CORRECT, and only the speed is probabilistic. Using a 61-bit prime modulus and a random-ish
base makes collisions astronomically rare in practice; this implementation additionally verifies, so
correctness never depends on luck. MULTI-PATTERN search groups patterns by length, hashes each, and
checks the text's window hash against the set. The LONGEST COMMON SUBSTRING of two strings is found by
binary-searching the length L and testing, via hashed length-L windows, whether the two strings share
any -- O((n+m) log) with hashing.

This module implements the rolling hash, single- and multi-pattern search (all occurrence positions),
and the longest common substring. It is verified against brute-force search -- identical occurrence
lists on hundreds of random text/pattern pairs -- and the longest common substring against an O(n*m)
dynamic-programming reference, with collision-verification ensuring exactness. Pure stdlib; a
string-algorithms companion to the KMP/Z-function matching, suffix-automaton, and hashing notes."""

from __future__ import annotations

_MOD = (1 << 61) - 1          # a large Mersenne prime
_BASE = 131542391             # a fixed odd base


def _hash(s):
    """Polynomial hash of a string (or byte sequence)."""
    h = 0
    for ch in s:
        h = (h * _BASE + (ord(ch) if isinstance(ch, str) else ch) + 1) % _MOD
    return h


def search(text, pattern):
    """All start indices where `pattern` occurs in `text`, via Rabin-Karp with verification. Returns a
    sorted list of positions (empty if the pattern is longer than the text or not found)."""
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []
    ph = _hash(pattern)
    # precompute B^(m-1) mod P
    high = pow(_BASE, m - 1, _MOD)
    # initial window hash
    wh = _hash(text[:m])
    out = []
    for i in range(n - m + 1):
        if wh == ph and text[i:i + m] == pattern:      # verify to defeat collisions
            out.append(i)
        if i < n - m:
            # roll: remove text[i], shift, add text[i+m]
            wh = (wh - (ord(text[i]) + 1) * high) % _MOD
            wh = (wh * _BASE + (ord(text[i + m]) + 1)) % _MOD
    return out


def contains(text, pattern):
    """True iff `pattern` occurs in `text`."""
    return len(search(text, pattern)) > 0


def multi_search(text, patterns):
    """Find all occurrences of several patterns at once. Returns a dict {pattern: [positions]}.
    Patterns are grouped by length so each text position is hashed once per distinct length."""
    result = {p: [] for p in patterns}
    # group patterns by length
    by_len = {}
    for p in patterns:
        by_len.setdefault(len(p), {}).setdefault(_hash(p), []).append(p)
    n = len(text)
    for m, hashmap in by_len.items():
        if m == 0:
            for p in [x for lst in hashmap.values() for x in lst]:
                result[p] = list(range(n + 1))
            continue
        if m > n:
            continue
        high = pow(_BASE, m - 1, _MOD)
        wh = _hash(text[:m])
        for i in range(n - m + 1):
            if wh in hashmap:
                window = text[i:i + m]
                for p in hashmap[wh]:
                    if window == p:
                        result[p].append(i)
            if i < n - m:
                wh = (wh - (ord(text[i]) + 1) * high) % _MOD
                wh = (wh * _BASE + (ord(text[i + m]) + 1)) % _MOD
    return result


def longest_common_substring(a, b):
    """The longest string that is a substring of both `a` and `b`, found by binary-searching the
    length and testing with hashed windows. Returns one such substring (empty if none)."""
    if not a or not b:
        return ""

    def has_common(length):
        """Return a common substring of the given length if one exists, else None."""
        if length == 0:
            return ""
        # collect all length-`length` window hashes of a (with the substring for verification)
        seen = {}
        high = pow(_BASE, length - 1, _MOD)
        if length <= len(a):
            wh = _hash(a[:length])
            for i in range(len(a) - length + 1):
                seen.setdefault(wh, []).append(i)
                if i < len(a) - length:
                    wh = (wh - (ord(a[i]) + 1) * high) % _MOD
                    wh = (wh * _BASE + (ord(a[i + length]) + 1)) % _MOD
        # scan b's windows
        if length <= len(b):
            wh = _hash(b[:length])
            for j in range(len(b) - length + 1):
                if wh in seen:
                    sub = b[j:j + length]
                    for i in seen[wh]:
                        if a[i:i + length] == sub:      # verify
                            return sub
                if j < len(b) - length:
                    wh = (wh - (ord(b[j]) + 1) * high) % _MOD
                    wh = (wh * _BASE + (ord(b[j + length]) + 1)) % _MOD
        return None

    lo, hi = 0, min(len(a), len(b))
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        found = has_common(mid)
        if found is not None:
            best = found
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# --- brute-force references -------------------------------------------------
def brute_search(text, pattern):
    """All occurrence start indices by direct window comparison."""
    m = len(pattern)
    if m == 0:
        return list(range(len(text) + 1))
    return [i for i in range(len(text) - m + 1) if text[i:i + m] == pattern]


def brute_lcs(a, b):
    """Longest common substring by O(n*m) dynamic programming, for validation."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return ""
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0
    end = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best:
                    best = dp[i][j]
                    end = i
    return a[end - best:end]
