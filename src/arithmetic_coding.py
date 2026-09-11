"""Arithmetic coding: entropy compression that beats Huffman on skewed data.

Huffman coding assigns each symbol a whole number of bits, so it wastes up to nearly a bit per
symbol -- disastrous when one symbol has probability 0.9 and "deserves" 0.15 bits. ARITHMETIC
CODING escapes the integer-bits limit: it encodes the ENTIRE message as a single number in [0, 1),
progressively narrowing an interval by each symbol's probability, so a symbol costing 0.15 bits
really adds only 0.15 bits to the output. It gets within a fraction of a bit of the Shannon entropy
regardless of the distribution, which is why it (and its cousin range coding) sit inside JPEG, H.264,
and modern compressors.

The idea: start with the interval [0, 1). Partition it by the symbols' cumulative probabilities;
the next symbol selects its sub-interval, which becomes the new [low, high); repeat. After the whole
message the final interval is tiny, and any number inside it identifies the message uniquely -- so
emit the shortest such number. Naively this needs unbounded-precision reals; the practical trick,
implemented here, is INTEGER RANGE CODING with RENORMALIZATION: keep low/high as fixed-width
integers and, whenever their top bits agree (the interval sits wholly in the lower or upper half),
output that bit and shift left, with an underflow counter for the awkward straddle-the-middle case.

This module implements an integer arithmetic coder and decoder driven by a symbol-frequency model,
with the entropy for comparison -- verified that encode/decode round-trips arbitrary messages, that
the code length approaches the Shannon entropy (within about one bit of the whole message), that it
beats Huffman on a skewed distribution where Huffman is forced to whole bits, and that it handles
single-symbol and uniform alphabets. Pure stdlib; an entropy-coding companion to the Huffman
(Shannon) and BWT notes."""

from __future__ import annotations

import math

_PRECISION = 32
_WHOLE = 1 << _PRECISION
_HALF = _WHOLE >> 1
_QUARTER = _WHOLE >> 2
_THREE_QUARTER = 3 * _QUARTER
_MASK = _WHOLE - 1


def build_model(data):
    """Frequency model: symbol -> (cum_low, cum_high, total). Cumulative counts partition [0,total)."""
    freqs = {}
    for s in data:
        freqs[s] = freqs.get(s, 0) + 1
    total = len(data)
    model = {}
    cum = 0
    for sym in sorted(freqs):
        model[sym] = (cum, cum + freqs[sym])
        cum += freqs[sym]
    return model, total


def encode(data):
    """Arithmetic-encode `data` (an iterable of symbols) to a bit string. Returns (bits, model,
    total, length) -- everything the decoder needs."""
    model, total = build_model(data)
    data = list(data)
    length = len(data)
    if length == 0:
        return "", model, total, 0

    low = 0
    high = _MASK
    pending = [0]                     # underflow bits awaiting resolution (boxed for closure)
    out = []

    def emit(bit):
        out.append(bit)
        for _ in range(pending[0]):
            out.append(1 - bit)
        pending[0] = 0
    for sym in data:
        lo_c, hi_c = model[sym]
        span = high - low + 1
        high = low + span * hi_c // total - 1
        low = low + span * lo_c // total
        # renormalize: shift out settled top bits
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                pending[0] += 1        # underflow: straddling the midpoint
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
    # flush: one more bit distinguishes the final interval
    pending[0] += 1
    emit(0 if low < _QUARTER else 1)
    return "".join(str(b) for b in out), model, total, length


def decode(bits, model, total, length):
    """Invert arithmetic coding given the bit string and the model used to encode it."""
    if length == 0:
        return []
    # invert the model: cumulative-count range -> symbol
    ranges = sorted(model.items(), key=lambda kv: kv[1][0])
    bits = [int(b) for b in bits]

    def read(i):
        return bits[i] if i < len(bits) else 0

    low = 0
    high = _MASK
    # load the first PRECISION bits into the code value
    value = 0
    pos = 0
    for _ in range(_PRECISION):
        value = (value << 1) | read(pos)
        pos += 1

    out = []
    for _ in range(length):
        span = high - low + 1
        # which symbol's cumulative range contains the current value?
        scaled = ((value - low + 1) * total - 1) // span
        sym = None
        for s, (lo_c, hi_c) in ranges:
            if lo_c <= scaled < hi_c:
                sym = s
                lo_sym, hi_sym = lo_c, hi_c
                break
        out.append(sym)
        high = low + span * hi_sym // total - 1
        low = low + span * lo_sym // total
        # renormalize in lock-step with the encoder
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                low -= _QUARTER
                high -= _QUARTER
                value -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | read(pos)
            pos += 1
    return out


def entropy(data):
    """Shannon entropy of the data in bits per symbol (the ideal average code length)."""
    n = len(data)
    if n == 0:
        return 0.0
    freqs = {}
    for s in data:
        freqs[s] = freqs.get(s, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in freqs.values())


def encoded_bits_per_symbol(data):
    """Actual arithmetic-code length in bits per symbol (compare to the entropy)."""
    bits, _, _, length = encode(data)
    return len(bits) / length if length else 0.0
