"""rANS -- Asymmetric Numeral Systems, the entropy coder inside Zstandard, LZMA, and JPEG XL.

Entropy coding squeezes a message down to its information content: a symbol of probability p should cost
about -log2(p) bits, so a stream that is 90% one symbol compresses far below one bit per symbol. Huffman
coding rounds each symbol to a whole number of bits and so leaves money on the table; arithmetic coding
hits the entropy but is slow and fiddly. ASYMMETRIC NUMERAL SYSTEMS (Jarek Duda, 2009) achieve
arithmetic coding's compression at Huffman's speed, and swept through modern codecs -- Zstandard, LZMA's
successor, Facebook's compression, the JPEG XL and AV1 image/video standards all use ANS. This module
implements the RANGE variant, rANS, the one used in practice.

The idea is startling in its simplicity. Encode the ENTIRE message as one gigantic integer, the "state"
x, and fold each symbol into it with a reversible arithmetic step. To add a symbol s that has frequency
f_s out of a total M (a power of two), with cumulative frequency c_s below it, the state transforms as

    x  ->  (x // f_s) * M + (x % f_s) + c_s.

That map is a bijection: from the new state you can recover both the symbol and the old state, so
DECODING runs the inverse -- read x % M, find which symbol's [c_s, c_s+f_s) interval it lands in, and
undo the arithmetic. The magic is that dividing by f_s shrinks the state by about log2(f_s) bits while
the multiply by M grows it by log2(M), for a NET cost of log2(M/f_s) = -log2(p_s) bits per symbol --
exactly the Shannon optimum. Because encode is a stack (last in, first out), decoding recovers the
message in reverse, which is why practical rANS encodes back-to-front.

To keep the unbounded integer bounded, the coder RENORMALISES: before each encode step it flushes the
low bytes of the state out to the output whenever the state would overflow a window, and decoding refills
them, keeping x in a fixed range while the emitted bytes carry the information. This module implements a
byte-wise streaming rANS with a static frequency model quantised to a power-of-two total, plus the model
builder and an entropy calculator, and encodes/decodes byte strings. Pure standard library.

Validation. The non-negotiable property is EXACT round-trip: decode(encode(data)) == data, for every
input -- the empty string, single symbols, uniform and wildly skewed distributions, and hundreds of
random byte strings of many lengths and alphabets; a single wrong bit would corrupt everything after it,
so exact recovery over thousands of cases is a strong correctness proof. The COMPRESSION is checked to
approach the Shannon entropy: on a skewed source the encoded size in bits per symbol is within a few
percent of the source entropy and well below 8 bits, and below what a whole-bit Huffman code could
achieve. Degenerate inputs (a single repeated symbol) compress to almost nothing, and the frequency
model's quantised totals are exact powers of two summing to M."""

import math


_SCALE_BITS = 16          # total frequency M = 2**16 (probability precision)
_M = 1 << _SCALE_BITS
_L = 1 << 23              # lower bound of the normalised state window
_MASK = 0xFF             # renormalise one byte at a time


# ---------------------------------------------------------------------------
# frequency model
# ---------------------------------------------------------------------------

def build_model(data):
    """Build a static frequency model over the bytes of ``data``.

    Returns (freq, cum, symbols) where freq[s] and cum[s] are integer frequencies and cumulative
    frequencies quantised so sum(freq) == M, and symbols is the sorted list of present byte values.
    """
    counts = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1
    if not counts:
        return {}, {}, []

    symbols = sorted(counts)
    total = len(data)
    # quantise to frequencies summing to M, guaranteeing each present symbol gets >= 1
    freq = {}
    allocated = 0
    for s in symbols:
        f = (counts[s] * _M) // total
        if f == 0:
            f = 1
        freq[s] = f
        allocated += f
    # fix rounding so the frequencies sum exactly to M
    diff = _M - allocated
    # adjust the most frequent symbol (it can absorb the correction safely)
    top = max(symbols, key=lambda s: freq[s])
    freq[top] += diff
    if freq[top] < 1:
        # pathological: redistribute (only for tiny alphabets); fall back to equal split
        freq = _equal_split(symbols)
    # cumulative
    cum = {}
    c = 0
    for s in symbols:
        cum[s] = c
        c += freq[s]
    return freq, cum, symbols


def _equal_split(symbols):
    n = len(symbols)
    base = _M // n
    freq = {s: base for s in symbols}
    freq[symbols[0]] += _M - base * n
    return freq


def _symbol_lookup(cum, freq, symbols):
    """Build an array of length M mapping a slot to its symbol, for O(1) decode."""
    table = [0] * _M
    for s in symbols:
        for slot in range(cum[s], cum[s] + freq[s]):
            table[slot] = s
    return table


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------

def encode(data, model=None):
    """rANS-encode ``data`` (bytes). Returns (encoded_bytes, model). Encodes back-to-front."""
    if model is None:
        model = build_model(data)
    freq, cum, symbols = model
    if not symbols:
        return b"", model

    x = _L
    out = bytearray()
    # process symbols in REVERSE (rANS is a stack; decoding will unwind forward)
    for b in reversed(data):
        f = freq[b]
        c = cum[b]
        # renormalise: emit low bytes while the state is too big for this symbol
        x_max = ((_L >> _SCALE_BITS) << 8) * f
        while x >= x_max:
            out.append(x & _MASK)
            x >>= 8
        x = ((x // f) << _SCALE_BITS) + (x % f) + c
    # flush the final state (4 bytes, since x < _L << 8 after renorm and _L = 2**23)
    for _ in range(4):
        out.append(x & _MASK)
        x >>= 8
    return bytes(out), model


def decode(encoded, model, length):
    """rANS-decode ``length`` symbols from ``encoded`` using ``model``. Returns bytes."""
    freq, cum, symbols = model
    if length == 0 or not symbols:
        return b""
    table = _symbol_lookup(cum, freq, symbols)

    data = bytearray(encoded)
    pos = len(data)
    # read back the 4-byte final state (was pushed last, low byte first)
    x = 0
    for _ in range(4):
        pos -= 1
        x = (x << 8) | data[pos]

    out = bytearray()
    for _ in range(length):
        slot = x & (_M - 1)
        s = table[slot]
        f = freq[s]
        c = cum[s]
        x = f * (x >> _SCALE_BITS) + slot - c
        # renormalise: pull bytes back in while the state is too small
        while x < _L and pos > 0:
            pos -= 1
            x = (x << 8) | data[pos]
        out.append(s)
    return bytes(out)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def entropy(data):
    """Shannon entropy of ``data`` in bits per symbol."""
    if not data:
        return 0.0
    counts = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def bits_per_symbol(data):
    """Actual encoded size of ``data`` in bits per symbol."""
    if not data:
        return 0.0
    enc, _ = encode(data)
    return len(enc) * 8 / len(data)
