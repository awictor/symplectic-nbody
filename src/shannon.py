"""Shannon entropy and Huffman coding: the limit of lossless compression.

How few bits does it take, on average, to record one symbol from a source that emits symbol i
with probability p_i? Shannon's answer (1948) is the entropy

    H = -sum_i p_i log2 p_i   bits per symbol,

and his source-coding theorem says no lossless code can beat it: the average codeword length L
of any uniquely decodable code satisfies L >= H, and there is always a code with L < H + 1. A
fair coin carries 1 bit; a biased or skewed source carries less, because the frequent symbols
are predictable. Entropy is maximal (log2 n bits) when all n symbols are equally likely and
zero when one symbol is certain -- it measures surprise.

Huffman's algorithm (1952) builds the optimal such code: repeatedly merge the two least
probable symbols into a subtree, and the resulting prefix code (no codeword is a prefix of
another, so the stream decodes unambiguously) achieves the smallest possible average length. It
provably lands in the [H, H+1) band and is exactly optimal among prefix codes. Every prefix
code obeys the Kraft inequality sum_i 2^{-len_i} <= 1, with equality for a complete code.

This module computes the entropy of a distribution, builds a Huffman code from symbol
frequencies, measures its average length and efficiency, verifies the Shannon bound and the
Kraft inequality, and round-trips encode/decode to prove the code is lossless. Pure stdlib; the
information-theory companion to the Benford and Bayes notes.
"""

from __future__ import annotations

import math


def entropy(probs) -> float:
    """Shannon entropy H = -sum p log2 p (bits) of a probability distribution. Zero-probability
    symbols contribute nothing (0 log 0 := 0)."""
    h = 0.0
    for p in probs:
        if p < 0:
            raise ValueError("probabilities must be nonnegative")
        if p > 0:
            h -= p * math.log2(p)
    return h


def entropy_from_counts(counts) -> float:
    """Entropy (bits/symbol) of an empirical distribution given integer symbol counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return entropy([c / total for c in counts])


def max_entropy(n: int) -> float:
    """Maximum possible entropy for n symbols: log2(n), attained by the uniform distribution."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    return math.log2(n)


def _normalize(freqs):
    """Accept a dict {symbol: weight} or a list of weights; return (symbols, probs)."""
    if isinstance(freqs, dict):
        symbols = list(freqs.keys())
        weights = [freqs[s] for s in symbols]
    else:
        symbols = list(range(len(freqs)))
        weights = list(freqs)
    total = sum(weights)
    if total <= 0:
        raise ValueError("frequencies must sum to a positive value")
    return symbols, [w / total for w in weights]


def huffman_code(freqs):
    """Build an optimal Huffman prefix code from symbol frequencies (a dict {symbol: weight} or
    a list of weights). Returns {symbol: bitstring}. A single symbol is coded as "0"."""
    symbols, probs = _normalize(freqs)
    if len(symbols) == 1:
        return {symbols[0]: "0"}
    # nodes: [weight, tiebreak_id, symbol_or_None, left, right]
    heap = [[probs[i], i, symbols[i], None, None] for i in range(len(symbols))]
    next_id = len(symbols)
    # simple heap-free approach: repeatedly pull the two smallest (n is small in practice)
    while len(heap) > 1:
        heap.sort(key=lambda node: (node[0], node[1]))
        a = heap.pop(0)
        b = heap.pop(0)
        merged = [a[0] + b[0], next_id, None, a, b]
        next_id += 1
        heap.append(merged)
    root = heap[0]
    codes = {}

    def walk(node, prefix):
        if node[2] is not None:  # leaf
            codes[node[2]] = prefix or "0"
            return
        walk(node[3], prefix + "0")
        walk(node[4], prefix + "1")

    walk(root, "")
    return codes


def average_length(code, freqs) -> float:
    """Expected codeword length sum_i p_i len(code_i) in bits per symbol."""
    symbols, probs = _normalize(freqs)
    p = {symbols[i]: probs[i] for i in range(len(symbols))}
    return sum(p[s] * len(code[s]) for s in code)


def efficiency(code, freqs) -> float:
    """Coding efficiency H / L in [0, 1]: how close the code comes to the entropy limit."""
    _, probs = _normalize(freqs)
    h = entropy(probs)
    L = average_length(code, freqs)
    return h / L if L > 0 else 1.0


def kraft_sum(code) -> float:
    """The Kraft-McMillan sum sum_i 2^{-len_i}. A valid prefix code has this <= 1 (= 1 when the
    code tree is complete/full)."""
    return sum(2.0 ** (-len(c)) for c in code.values())


def is_prefix_free(code) -> bool:
    """True if no codeword is a prefix of another (the property that makes the code decodable)."""
    words = sorted(code.values(), key=len)
    for i, w in enumerate(words):
        for longer in words[i + 1:]:
            if longer.startswith(w) and longer != w:
                return False
    return True


def encode(message, code) -> str:
    """Encode an iterable of symbols to a bitstring using the code."""
    return "".join(code[s] for s in message)


def decode(bits: str, code):
    """Decode a bitstring back to the list of symbols (walks the prefix code)."""
    inv = {v: k for k, v in code.items()}
    out = []
    cur = ""
    for bit in bits:
        cur += bit
        if cur in inv:
            out.append(inv[cur])
            cur = ""
    if cur:
        raise ValueError("bitstring is not a whole number of codewords")
    return out
