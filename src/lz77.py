"""LZ77: compression by pointing back at what you have already seen.

Abraham Lempel and Jacob Ziv's 1977 algorithm is the engine inside ZIP, gzip, PNG, and almost
every general-purpose compressor. Its idea is disarmingly simple: as you scan the data, whenever
the next few bytes have already appeared recently, do not repeat them -- emit a back-reference, a
pair (distance, length) that says "copy `length` bytes from `distance` positions back". Only
genuinely new bytes are stored literally.

The compressor keeps a sliding WINDOW of the most recent bytes (the dictionary) and a small
look-ahead buffer, and at each step finds the longest match for the look-ahead inside the window.
It emits a token: either a literal byte, or a (distance, length) copy followed by the next
literal. The decompressor replays the tokens, copying from its own growing output -- so it needs
no dictionary of its own, and the copy can even overlap its source (that is how a run of a single
byte compresses to one token).

Compression comes for free on any data with repetition: English text, source code, logs, genomes.
Random data cannot be compressed (a token costs more than the bytes it saves), which is exactly
Shannon's limit showing through -- you cannot beat the entropy. This module implements the
sliding-window encoder and the replay decoder, measures the compression ratio and token count,
and guarantees a lossless round-trip. Pure stdlib; the dictionary-coding companion to the
Shannon-entropy and Huffman note (LZ77 + Huffman together are DEFLATE, the heart of gzip).
"""

from __future__ import annotations


def compress(data: bytes, window: int = 4096, look_ahead: int = 255):
    """Compress bytes with LZ77. Returns a list of tokens, each either:

        ("lit", byte)                      a literal byte, or
        ("copy", distance, length, byte)   copy `length` bytes from `distance` back, then a
                                           literal `byte` (or None at end of stream).

    `window` bounds how far back a match may point; `look_ahead` bounds the match length.
    """
    tokens = []
    i = 0
    n = len(data)
    while i < n:
        best_len = 0
        best_dist = 0
        start = max(0, i - window)
        # search the window for the longest prefix of data[i:] that appears starting in [start, i)
        for j in range(start, i):
            length = 0
            while (length < look_ahead and i + length < n
                   and data[j + length] == data[i + length]):
                length += 1
            if length > best_len:
                best_len = length
                best_dist = i - j
        if best_len >= 3:  # only worth a copy token if it saves space
            nxt = data[i + best_len] if i + best_len < n else None
            tokens.append(("copy", best_dist, best_len, nxt))
            i += best_len + (1 if nxt is not None else 0)
        else:
            tokens.append(("lit", data[i]))
            i += 1
    return tokens


def decompress(tokens) -> bytes:
    """Replay LZ77 tokens back into the original bytes. Handles overlapping copies (a copy whose
    length exceeds its distance, used to expand runs)."""
    out = bytearray()
    for token in tokens:
        if token[0] == "lit":
            out.append(token[1])
        else:  # copy
            _, dist, length, nxt = token
            start = len(out) - dist
            if start < 0:
                raise ValueError("copy distance points before the start of the stream")
            for k in range(length):
                out.append(out[start + k])  # overlap-safe: reads bytes just written
            if nxt is not None:
                out.append(nxt)
    return bytes(out)


def token_cost(tokens, literal_bits: int = 8, copy_bits: int = 24) -> int:
    """Estimated compressed size in bits: literals cost `literal_bits`, copy tokens cost
    `copy_bits` (a distance+length+flag). A simple model of the serialized stream size."""
    bits = 0
    for token in tokens:
        if token[0] == "lit":
            bits += literal_bits
        else:
            bits += copy_bits + (literal_bits if token[3] is not None else 0)
    return bits


def compression_ratio(data: bytes, tokens=None, **kw) -> float:
    """Ratio of original size to estimated compressed size (higher = better compression). A
    ratio of 1.0 means no gain; below 1.0 means the tokens cost more than the raw bytes."""
    if tokens is None:
        tokens = compress(data, **kw)
    original_bits = len(data) * 8
    comp_bits = token_cost(tokens)
    return original_bits / comp_bits if comp_bits else float("inf")


def count_tokens(tokens):
    """Return (literals, copies) counts."""
    lits = sum(1 for t in tokens if t[0] == "lit")
    copies = sum(1 for t in tokens if t[0] == "copy")
    return lits, copies
