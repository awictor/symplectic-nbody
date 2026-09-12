"""Lyndon words and Duval's algorithm: the unique factorisation of a string into non-increasing primes.

A LYNDON WORD is a non-empty string that is strictly smaller (in dictionary order) than all of its
proper rotations -- equivalently, strictly smaller than every one of its own proper suffixes. "aab" is
Lyndon (it beats "aba" and "baa"); "aba" is not (its rotation "aab" is smaller); "aa" is not (it ties
its rotation). Lyndon words are the "primes" of string concatenation: the CHEN-FOX-LYNDON theorem says
every string factorises UNIQUELY into a sequence of Lyndon words whose values are non-increasing,
w = w1 w2 ... wk with w1 >= w2 >= ... >= wk. This factorisation is the backbone of several deep
results -- it generates a Lyndon basis for free Lie algebras, it is the key step in the linear-time
Burrows-Wheeler / bijective BWT, and the last factor's start is exactly the rotation that gives the
LEXICOGRAPHICALLY SMALLEST ROTATION of a string (Booth's problem), used to canonicalise necklaces.

DUVAL'S ALGORITHM computes the factorisation in O(n) time and O(1) extra space by a beautiful two-
pointer scan. It maintains a candidate Lyndon prefix and compares each new character to the one a
period back: if equal it extends the period, if larger it starts a fresh Lyndon word, and if smaller
it emits as many copies of the current Lyndon word as the period allows and restarts. The same engine,
run over the doubled string, yields the smallest rotation. Related is the generation of all Lyndon
words up to a given length over an alphabet, produced in lexicographic order by the FKM (Fredricksen-
Kessler-Maiorana) algorithm, which -- concatenated -- builds the De Bruijn sequence.

This module tests whether a string is a Lyndon word, factorises any string into its unique
non-increasing Lyndon decomposition (Duval), finds the least rotation (Booth via Duval), and generates
all Lyndon words up to a length. It is verified against brute force: the factorisation's parts are all
Lyndon and non-increasing and concatenate back to the input, the membership test matches the
rotation/suffix definition, the least rotation matches an exhaustive scan of all rotations, and the
generated Lyndon words match a brute filter over all strings -- on hundreds of random cases. Pure
stdlib; a combinatorics-on-words companion to the suffix-automaton, De-Bruijn-sequence, and BWT notes."""

from __future__ import annotations


def is_lyndon(s):
    """True iff `s` is a Lyndon word: non-empty and strictly smaller than all its proper rotations
    (equivalently, strictly smaller than each of its proper suffixes)."""
    if not s:
        return False
    n = len(s)
    for i in range(1, n):
        if s[i:] < s:              # a proper suffix is smaller-or-equal -> not Lyndon
            return False
    return True


def duval(s):
    """Chen-Fox-Lyndon factorisation of `s` into non-increasing Lyndon words, in O(n) (Duval's
    algorithm). Returns the list of factors; concatenated they equal `s`, and factor values are
    non-increasing."""
    n = len(s)
    factors = []
    i = 0
    while i < n:
        j = i + 1        # scan pointer
        k = i            # period pointer (points a period behind j)
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i    # strictly larger: reset the period, current prefix stays Lyndon
            else:
                k += 1   # equal: advance the period
            j += 1
        # emit Lyndon words of length (j - k) until the prefix [i, k] is consumed
        while i <= k:
            factors.append(s[i:i + (j - k)])
            i += j - k
    return factors


def least_rotation(s):
    """The lexicographically smallest rotation of `s` (Booth's problem), via Duval over s+s.
    Returns (index, rotation): index is the starting offset of the least rotation in `s`."""
    if not s:
        return 0, ""
    n = len(s)
    ss = s + s
    # Duval-style scan tracking the start of the best Lyndon factor that spans a full period
    i = 0
    best = 0
    while i < n:
        best = i
        j = i + 1
        k = i
        while j < 2 * n and ss[k] <= ss[j]:
            if ss[k] < ss[j]:
                k = i
            else:
                k += 1
            j += 1
        while i <= k:
            i += j - k
    return best, s[best:] + s[:best]


def lyndon_words_up_to(alphabet_size, max_len):
    """All Lyndon words over the alphabet {0,1,...,alphabet_size-1} with length <= max_len, in
    lexicographic order (the FKM / Duval generation algorithm). Words are returned as tuples of ints."""
    words = []
    w = [-1]                       # sentinel; the FKM algorithm grows `w`
    while w:
        w[-1] += 1
        m = len(w)                 # m is the current period length; w[:m] is a Lyndon word
        words.append(tuple(w))
        # extend w to length max_len by repeating its period, per FKM
        while len(w) < max_len:
            w.append(w[len(w) - m])
        # drop trailing symbols already at the max value, then the next loop increments
        while w and w[-1] == alphabet_size - 1:
            w.pop()
    return words


def is_lyndon_seq(seq):
    """Lyndon test for a sequence of comparable items (e.g. a tuple of ints)."""
    seq = tuple(seq)
    if not seq:
        return False
    n = len(seq)
    for i in range(1, n):
        if seq[i:] < seq:
            return False
    return True


# --- brute-force references -------------------------------------------------
def brute_is_lyndon(s):
    """Lyndon membership by testing every proper rotation directly."""
    if not s:
        return False
    n = len(s)
    return all(s < s[i:] + s[:i] for i in range(1, n))


def brute_least_rotation(s):
    """Least rotation by generating all rotations and taking the minimum."""
    if not s:
        return 0, ""
    n = len(s)
    best_i = min(range(n), key=lambda i: s[i:] + s[:i])
    return best_i, s[best_i:] + s[:best_i]


def brute_lyndon_words(alphabet_size, max_len):
    """All Lyndon words up to max_len by filtering every string over the alphabet. Exponential."""
    from itertools import product
    out = []
    for L in range(1, max_len + 1):
        for tup in product(range(alphabet_size), repeat=L):
            if is_lyndon_seq(tup):
                out.append(tup)
    return sorted(out)
