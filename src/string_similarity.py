"""String similarity: Jaro-Winkler, q-gram Jaccard/Dice, and Soundex phonetic matching.

Comparing strings is not one problem but several. EDIT DISTANCE counts single-character
insert/delete/substitute operations -- good for typos, but O(nm) and blind to how humans actually
mistype. This module collects the other workhorses of fuzzy matching, each capturing a different
notion of "close":

  JARO similarity counts MATCHING characters (within a sliding window) and TRANSPOSITIONS, giving a
      score in [0,1] tuned for short strings like names. JARO-WINKLER boosts it when the strings
      share a common PREFIX -- the observation that people rarely mistype the first few letters, so
      "Martha"/"Marhta" scores very high. It is the classic method for record linkage and
      deduplicating name lists.
  Q-GRAM overlap treats each string as its bag of length-q substrings and measures set overlap by
      the JACCARD index |A n B| / |A u B| or the SORENSEN-DICE coefficient 2|A n B| / (|A| + |B|).
      Robust to word reordering and good for longer text.
  SOUNDEX is a PHONETIC code: it maps a word to a letter-plus-three-digits code so that words that
      SOUND alike collide ("Robert" and "Rupert" both -> R163). It powers "sounds-like" search and
      genealogical name matching.

This module implements all of these plus a small demo of ranking candidates against a query.
Validated: Jaro and Jaro-Winkler reproduce the textbook values (Martha/Marhta = 0.944 Jaro, 0.961
Winkler), identical strings score 1 and disjoint strings 0, Winkler is at least Jaro and boosts only
on a shared prefix; the Jaccard/Dice coefficients are symmetric, 1 for identical, 0 for disjoint, and
satisfy Dice >= Jaccard; and Soundex reproduces the standard reference codes (Robert->R163,
Rupert->R163, Tymczak->T522) and its homophones collide. Pure stdlib; the fuzzy-matching companion to
the Levenshtein edit distance and the n-gram / hashing tools."""

from __future__ import annotations

import math


def jaro(s1, s2):
    """Jaro similarity in [0, 1]: matching characters within a window, minus transpositions."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    # count transpositions
    t = 0
    k = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
    t //= 2
    m = matches
    return (m / len1 + m / len2 + (m - t) / m) / 3


def jaro_winkler(s1, s2, p=0.1, max_prefix=4):
    """Jaro-Winkler: boost Jaro by a common-prefix bonus (up to max_prefix chars, scale p<=0.25)."""
    j = jaro(s1, s2)
    # common prefix length
    prefix = 0
    for a, b in zip(s1, s2):
        if a == b:
            prefix += 1
            if prefix == max_prefix:
                break
        else:
            break
    return j + prefix * p * (1 - j)


def qgrams(s, q=2, pad=True):
    """The multiset of length-q substrings of s (padded with sentinels so short strings still have
    q-grams). Returns a set (for Jaccard/Dice)."""
    if pad:
        s = "#" * (q - 1) + s + "$" * (q - 1)
    if len(s) < q:
        return {s}
    return {s[i:i + q] for i in range(len(s) - q + 1)}


def jaccard(s1, s2, q=2):
    """Jaccard index of the q-gram sets: |A n B| / |A u B|."""
    a = qgrams(s1, q)
    b = qgrams(s2, q)
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def dice(s1, s2, q=2):
    """Sorensen-Dice coefficient of the q-gram sets: 2|A n B| / (|A| + |B|)."""
    a = qgrams(s1, q)
    b = qgrams(s2, q)
    if not a and not b:
        return 1.0
    inter = len(a & b)
    denom = len(a) + len(b)
    return 2 * inter / denom if denom else 1.0


_SOUNDEX_MAP = {}
for letters, digit in [("BFPV", "1"), ("CGJKQSXZ", "2"), ("DT", "3"),
                       ("L", "4"), ("MN", "5"), ("R", "6")]:
    for ch in letters:
        _SOUNDEX_MAP[ch] = digit


def soundex(word):
    """The Soundex phonetic code: first letter + three digits, so homophones collide (Robert->R163)."""
    word = "".join(c for c in word.upper() if c.isalpha())
    if not word:
        return ""
    first = word[0]
    # encode, collapsing adjacent duplicate codes and skipping vowels/H/W as separators
    codes = [_SOUNDEX_MAP.get(c, "") for c in word]
    out = first
    prev = _SOUNDEX_MAP.get(first, "")
    for i in range(1, len(word)):
        c = word[i]
        code = _SOUNDEX_MAP.get(c, "")
        if code:
            if code != prev:
                out += code
            prev = code
        else:
            # vowel, H or W: H and W do not reset the previous code; vowels do
            if c not in "HW":
                prev = ""
        if len(out) == 4:
            break
    return (out + "000")[:4]


def rank(query, candidates, method="jaro_winkler"):
    """Rank candidate strings by similarity to the query (descending). method in
    {jaro, jaro_winkler, jaccard, dice}."""
    fns = {"jaro": jaro, "jaro_winkler": jaro_winkler, "jaccard": jaccard, "dice": dice}
    fn = fns[method]
    scored = [(c, fn(query, c)) for c in candidates]
    scored.sort(key=lambda kv: -kv[1])
    return scored
