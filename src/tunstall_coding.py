"""Tunstall coding: the mirror image of Huffman -- map VARIABLE-length source strings to FIXED-length codes.

Huffman coding reads one symbol at a time and emits a variable number of bits: frequent symbols get short
codes, rare ones long. TUNSTALL CODING (1967) is its exact dual. It reads a VARIABLE number of source
symbols and emits a FIXED number of bits: it builds a dictionary of variable-length source strings, all
assigned equal-length codewords, chosen so that each dictionary entry is about equally probable. Because
every codeword is the same width, Tunstall codes are trivially synchronizable and byte-aligned -- prized
in systems where a fixed output rate or random access matters (packetized links, database columns), where
Huffman's bit-ragged output is awkward.

The construction is a greedy tree. Start with the source alphabet as the leaves of a tree. Repeatedly take
the MOST PROBABLE leaf and EXPAND it -- replace it by its children, one per source symbol, whose
probabilities are the parent's times each symbol's probability. Stop when the number of leaves would
exceed 2^k (the codeword budget for k-bit codes). The leaves are the dictionary strings; assign each a
distinct k-bit codeword. Expanding the most probable leaf equalizes the leaf probabilities, which is
exactly what maximizes the expected number of source symbols consumed per codeword -- and by a Tunstall
analogue of the source-coding theorem, the resulting bits-per-symbol approaches the source entropy from
above as k grows.

This module builds a Tunstall dictionary for a known symbol distribution, encodes a message into
fixed-width codewords (greedily matching the longest dictionary string at each step), decodes them back,
and reports the expected codeword length and the compression rate in bits per source symbol. It is
validated: encode/decode round-trips exactly; every dictionary string maps to a distinct k-bit codeword
and the dictionary is prefix-free over source strings; the achieved bits-per-symbol lies between the
source entropy and entropy+1 and TIGHTENS toward the entropy as k grows; a uniform source gives a balanced
dictionary of equal-length strings; the most-probable-leaf expansion equalizes leaf probabilities; and
results are deterministic. Pure stdlib; the variable-to-fixed-length companion to the Huffman,
arithmetic-coding, and Shannon-entropy tools."""

from __future__ import annotations

import heapq
import math


def build_dictionary(probs, k):
    """Build a Tunstall dictionary of source strings for k-bit codewords (at most 2^k entries).

    probs is a dict {symbol: probability} (need not be normalized). Returns a list of (string, prob)
    dictionary entries -- the leaves of the Tunstall tree -- sorted for determinism."""
    total = sum(probs.values())
    p = {s: probs[s] / total for s in probs}
    alphabet = sorted(p)
    m = len(alphabet)
    if m < 2:
        raise ValueError("need at least 2 source symbols")
    budget = 1 << k
    if budget < m:
        raise ValueError(f"k={k} too small: 2^k={budget} < alphabet size {m}")

    # leaves as a max-heap by probability (negate for heapq). Tie-break by string for determinism.
    # each leaf: (-prob, string)
    leaves = [(-p[s], s) for s in alphabet]
    heapq.heapify(leaves)

    # each expansion replaces 1 leaf with m leaves: net +(m-1). Stop before exceeding budget.
    while len(leaves) + (m - 1) <= budget:
        neg_prob, string = heapq.heappop(leaves)
        parent_prob = -neg_prob
        for s in alphabet:
            child = string + s
            heapq.heappush(leaves, (-(parent_prob * p[s]), child))

    entries = [(string, -neg_prob) for neg_prob, string in leaves]
    entries.sort(key=lambda e: (-e[1], e[0]))   # by descending prob, then string
    return entries


def assign_codewords(entries, k):
    """Assign each dictionary string a distinct k-bit codeword. Returns (str->code, code->str)."""
    enc = {}
    dec = {}
    for i, (string, _prob) in enumerate(entries):
        code = format(i, f"0{k}b")
        enc[string] = code
        dec[code] = string
    return enc, dec


def _match_leaf(message, i, enc, pad_symbol):
    """Walk forward from position i until the accumulated string is a dictionary leaf.

    The Tunstall dictionary is a COMPLETE tree over the alphabet, so any forward sequence reaches a
    leaf. If the message ends first (a trailing internal-node prefix), extend with pad_symbol until a
    leaf is reached -- the decoder trims the padded symbols using the transmitted length. Returns
    (leaf_string, next_i)."""
    n = len(message)
    s = ""
    j = i
    while s not in enc:
        if j < n:
            s += message[j]
            j += 1
        else:
            s += pad_symbol      # pad past end-of-message to complete a leaf
    return s, j


def encode(message, enc):
    """Encode a message (string over the source alphabet) into concatenated k-bit codewords.

    Parses the message into dictionary strings (walking the complete Tunstall tree) and emits each
    string's fixed-width codeword. The final partial string is padded to a leaf; decode() trims it
    back using the original length. Returns the bit string."""
    # the pad symbol is the most probable single symbol (shortest, most-frequent leaf's first char)
    pad_symbol = min(enc, key=lambda s: (len(s), s))[0]
    out = []
    i = 0
    n = len(message)
    while i < n:
        string, i = _match_leaf(message, i, enc, pad_symbol)
        out.append(enc[string])
    return "".join(out)


def decode(bits, dec, k, length=None):
    """Decode concatenated k-bit codewords back to the source message.

    If `length` (the original symbol count) is given, the decoded string is truncated to it, removing
    any symbols the encoder padded to complete the final leaf."""
    out = []
    for j in range(0, len(bits), k):
        code = bits[j:j + k]
        if len(code) < k:
            break  # trailing bit padding
        out.append(dec[code])
    s = "".join(out)
    return s[:length] if length is not None else s


def expected_length(entries):
    """Expected number of SOURCE symbols consumed per codeword = sum p_i * len(string_i)."""
    return sum(prob * len(string) for string, prob in entries)


def bits_per_symbol(entries, k):
    """Compression rate: k bits per codeword / expected source symbols per codeword."""
    return k / expected_length(entries)


def source_entropy(probs):
    """Shannon entropy of the source in bits per symbol."""
    total = sum(probs.values())
    h = 0.0
    for s in probs:
        p = probs[s] / total
        if p > 0:
            h -= p * math.log2(p)
    return h
