"""Huffman coding: the provably optimal prefix-free code, and its canonical form.

Given symbols with frequencies, a PREFIX CODE assigns each a bit-string so that no codeword is a
prefix of another -- which makes decoding unambiguous, reading left to right with no delimiters. The
cost of a code is the expected length, sum of frequency times codeword length. Huffman's 1952
algorithm builds the code of MINIMUM expected length by a greedy merge: repeatedly take the two
least-frequent symbols (or merged nodes), make them siblings under a new node whose frequency is
their sum, and push it back. The tree grows bottom-up; the two rarest symbols end up deepest, the
common ones shallow. The greedy choice is provably optimal by an exchange argument -- in any optimal
tree the two rarest symbols can be moved to be sibling leaves at maximum depth without increasing
cost -- so the merge never sacrifices optimality.

Huffman's cost obeys the tight bracket H <= L < H + 1, where H is the Shannon ENTROPY: the optimal
code is never shorter than entropy (Shannon's source-coding bound) and never wastes a full extra bit
per symbol. It matches entropy exactly when every frequency is a power of one-half.

This module builds the code, computes expected length and the compression ratio against a fixed-width
code, and produces the CANONICAL Huffman code -- the same codeword LENGTHS, but codewords assigned in
a standard order (by length then symbol) so the decoder needs only the length table, not the tree.
It encodes and decodes bitstreams and round-trips them.

Validated by optimality and correctness: the code is prefix-free and round-trips any message; the
expected length equals a brute-force search over all possible code-length assignments (via the
Kraft-inequality enumeration) for small alphabets; the H <= L < H+1 entropy bracket holds; canonical
and tree codes have identical lengths; and dyadic distributions hit entropy exactly. Pure stdlib; the
optimal-prefix-code companion to the arithmetic coder and rANS already in the repo."""

from __future__ import annotations

import heapq
import math
from collections import Counter


def build_code(freqs):
    """Build a Huffman code from a dict {symbol: frequency}. Returns {symbol: codeword string}.
    A single distinct symbol gets the codeword '0'."""
    if not freqs:
        return {}
    if len(freqs) == 1:
        (sym,) = freqs
        return {sym: "0"}

    # heap of (freq, tiebreak, node); node is either a symbol leaf or an internal (left, right)
    heap = []
    for i, (sym, f) in enumerate(sorted(freqs.items(), key=lambda kv: (kv[1], _key(kv[0])))):
        heapq.heappush(heap, (f, i, ("leaf", sym)))
    counter = len(heap)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, counter, ("node", n1, n2)))
        counter += 1
    _, _, root = heap[0]

    code = {}

    def walk(node, prefix):
        if node[0] == "leaf":
            code[node[1]] = prefix or "0"
            return
        walk(node[1], prefix + "0")
        walk(node[2], prefix + "1")

    walk(root, "")
    return code


def _key(sym):
    """Sort key that works for mixed symbol types (str/int) deterministically."""
    return (str(type(sym)), sym)


def code_lengths(freqs):
    """The codeword length assigned to each symbol."""
    return {sym: len(cw) for sym, cw in build_code(freqs).items()}


def expected_length(freqs):
    """Expected codeword length (bits per symbol) under the symbol frequencies."""
    total = sum(freqs.values())
    if total == 0:
        return 0.0
    code = build_code(freqs)
    return sum(freqs[s] * len(code[s]) for s in freqs) / total


def entropy(freqs):
    """Shannon entropy of the frequency distribution, in bits."""
    total = sum(freqs.values())
    if total == 0:
        return 0.0
    h = 0.0
    for f in freqs.values():
        if f > 0:
            p = f / total
            h -= p * math.log2(p)
    return h


def canonical_code(freqs):
    """Canonical Huffman code: same lengths as build_code, but codewords assigned in canonical order
    (sort by (length, symbol), assign consecutive integers, left-shifting when the length grows).
    Returns {symbol: codeword string}. The decoder can be rebuilt from lengths alone."""
    lengths = code_lengths(freqs)
    if not lengths:
        return {}
    if len(lengths) == 1:
        (sym,) = lengths
        return {sym: "0"}
    # order by length then symbol
    order = sorted(lengths, key=lambda s: (lengths[s], _key(s)))
    code = {}
    prev_len = None
    val = 0
    for sym in order:
        L = lengths[sym]
        if prev_len is None:
            val = 0
        else:
            val = (val + 1) << (L - prev_len)
        code[sym] = format(val, "0{}b".format(L))
        prev_len = L
    return code


def encode(message, code):
    """Encode a sequence of symbols into a bit-string using the given code."""
    return "".join(code[s] for s in message)


def decode(bits, code):
    """Decode a bit-string using the given prefix code. Raises ValueError on malformed input."""
    inv = {cw: s for s, cw in code.items()}
    out = []
    cur = ""
    for b in bits:
        cur += b
        if cur in inv:
            out.append(inv[cur])
            cur = ""
    if cur:
        raise ValueError("bit-string not fully decodable with this code")
    return out


def is_prefix_free(code):
    """True if no codeword is a prefix of another."""
    words = sorted(code.values(), key=len)
    for i, w in enumerate(words):
        for longer in words[i + 1:]:
            if longer.startswith(w) and longer != w:
                return False
    return True


def compression_ratio(freqs):
    """Ratio of a fixed-width code's size to Huffman's size (>1 means Huffman is smaller)."""
    n = len(freqs)
    if n <= 1:
        return 1.0
    fixed = math.ceil(math.log2(n))
    return fixed / expected_length(freqs)


# --- brute-force optimality reference ----------------------------------------
def brute_optimal_expected_length(freqs):
    """The minimum expected length over ALL valid prefix codes, found by enumerating length
    assignments that satisfy the Kraft inequality (sum 2^-len <= 1). Small alphabets only."""
    syms = list(freqs)
    n = len(syms)
    total = sum(freqs.values())
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    max_len = n  # a prefix code on n symbols never needs a length > n-1, but n is a safe cap
    best = [float("inf")]

    def kraft_ok(lengths):
        return sum(2 ** -L for L in lengths) <= 1.0 + 1e-12

    def rec(i, lengths):
        if i == n:
            if kraft_ok(lengths):
                cost = sum(freqs[syms[j]] * lengths[j] for j in range(n)) / total
                best[0] = min(best[0], cost)
            return
        # prune: even with the shallowest possible remaining lengths, must stay under bound
        for L in range(1, max_len + 1):
            lengths.append(L)
            # partial Kraft prune
            if sum(2 ** -x for x in lengths) <= 1.0 + 1e-12:
                rec(i + 1, lengths)
            lengths.pop()

    rec(0, [])
    return best[0]
