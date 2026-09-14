"""Golomb coding: the optimal prefix code for geometrically-distributed integers -- run lengths, gaps, residuals.

When you encode nonnegative integers whose probability DECAYS geometrically -- run lengths in a bilevel
image, gaps between set bits in a sparse index, prediction residuals in lossless audio -- Huffman needs an
explicit code table and arithmetic coding is heavier than necessary. GOLOMB CODING (1966) is the tailored
answer: a parametric prefix code that is PROVABLY OPTIMAL for a geometric source, needing only one integer
parameter m. It splits each value n into a quotient and remainder by m:

    q = n // m   ->  encoded in UNARY  (q ones then a zero),
    r = n %  m   ->  encoded in TRUNCATED BINARY  (ceil(log2 m) or floor(log2 m) bits).

Small values -- the common ones under a geometric law -- get short codes; large values pay a long unary
prefix, which is rare. The parameter m tunes the code to the decay rate: for a geometric distribution
P(n) = (1-p) p^n, the optimal m is about ceil( -1 / log2 p ), balancing the unary and binary parts. When m
is a power of two the remainder is plain binary and the scheme becomes RICE CODING, the fast special case
used in FLAC, Shorten, and JPEG-LS.

This module encodes and decodes single integers and streams with any m (truncated-binary remainders so no
codeword is wasted), computes the optimal m for a geometric source, and reports the mean code length. It
is validated: encode/decode round-trips for every m and every value; the codes are prefix-free (a stream
decodes without delimiters); the truncated-binary remainder uses the minimal bits and stays prefix-free;
the optimal-m code length is within a fraction of a bit of the source entropy on a geometric source and
beats a fixed-length code; Rice coding (m a power of two) agrees with Golomb; larger m shortens big values
and lengthens small ones (the tradeoff); and results are deterministic. Pure stdlib; the
geometric-source entropy-coding companion to the Huffman, Tunstall, arithmetic-coding, and Elias tools."""

from __future__ import annotations

import math


def _unary(q):
    """q ones followed by a terminating zero."""
    return "1" * q + "0"


def _truncated_binary(r, m):
    """Truncated binary code for r in [0, m): the first 2^(k+1)-m values use k bits, the rest k+1.

    This is the minimal-length prefix-free binary code for an m-ary alphabet (optimal when m is not a
    power of two). k = floor(log2 m)."""
    if m == 1:
        return ""
    k = m.bit_length() - 1          # floor(log2 m)
    cutoff = (1 << (k + 1)) - m     # first `cutoff` symbols get k bits
    if r < cutoff:
        return format(r, f"0{k}b")
    else:
        return format(r + cutoff, f"0{k + 1}b")


def _decode_truncated_binary(bits, i, m):
    """Read a truncated-binary remainder starting at bits[i]. Returns (r, next_i)."""
    if m == 1:
        return 0, i
    k = m.bit_length() - 1
    cutoff = (1 << (k + 1)) - m
    # read k bits
    first = int(bits[i:i + k], 2) if k > 0 else 0
    if first < cutoff:
        return first, i + k
    # read one more bit
    val = int(bits[i:i + k + 1], 2)
    return val - cutoff, i + k + 1


def encode_int(n, m):
    """Golomb-encode a single nonnegative integer n with parameter m. Returns a bit string."""
    if n < 0:
        raise ValueError("Golomb coding is for nonnegative integers")
    if m < 1:
        raise ValueError("m must be >= 1")
    q = n // m
    r = n % m
    return _unary(q) + _truncated_binary(r, m)


def decode_int(bits, m, i=0):
    """Decode one Golomb codeword from bits starting at index i. Returns (value, next_i)."""
    # read the unary quotient: count 1s until a 0
    q = 0
    while bits[i] == "1":
        q += 1
        i += 1
    i += 1  # skip the terminating 0
    r, i = _decode_truncated_binary(bits, i, m)
    return q * m + r, i


def encode(values, m):
    """Encode a list of nonnegative integers into one concatenated bit string."""
    return "".join(encode_int(n, m) for n in values)


def decode(bits, m, count):
    """Decode `count` Golomb codewords from a bit string."""
    out = []
    i = 0
    for _ in range(count):
        v, i = decode_int(bits, m, i)
        out.append(v)
    return out


def optimal_m(p):
    """Optimal Golomb parameter m for a geometric source P(n) = (1-p) p^n, 0 < p < 1.

    m = max(1, ceil( -1 / log2(p) ))  -- the standard closed form. Larger p (slower decay) -> larger m."""
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    return max(1, math.ceil(-1.0 / math.log2(p)))


def rice_m(k):
    """Rice parameter: m = 2^k (the power-of-two Golomb special case with plain binary remainders)."""
    return 1 << k


def mean_code_length(values, m):
    """Average Golomb code length in bits over a list of values."""
    total = sum(len(encode_int(n, m)) for n in values)
    return total / len(values)


def geometric_entropy(p):
    """Entropy (bits/symbol) of the geometric distribution P(n) = (1-p) p^n."""
    # H = -( (1-p) log2(1-p) + p log2 p ) / (1-p)
    return -((1 - p) * math.log2(1 - p) + p * math.log2(p)) / (1 - p)
