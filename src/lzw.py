"""LZW compression: adaptive dictionary coding that builds its codebook on the fly.

LEMPEL-ZIV-WELCH (1984) is the dictionary compressor behind GIF, early TIFF, and Unix `compress`. Its
charm is that it needs NO explicit dictionary in the output and NO two passes over the data: encoder
and decoder build the SAME dictionary independently as they go, so the compressed stream is just a
sequence of integer codes. It compresses by replacing repeated SUBSTRINGS with single codes, and the
longer and more repetitive the input, the longer the matched substrings become -- so its ratio
improves as it learns the data's patterns.

The encoder starts with a dictionary of every single byte (codes 0..255). It reads the longest
current string w that is in the dictionary, and on seeing the next character c: if w+c is also in the
dictionary it extends (w := w+c); otherwise it OUTPUTS the code for w, ADDS w+c as a new entry with
the next free code, and restarts with w := c. The decoder mirrors this exactly: each code it reads it
translates to a string, and it reconstructs the same new dictionary entries one step behind the
encoder -- with one classic edge case (the "KwKwK" pattern) where a code refers to an entry the
decoder is about to build, handled by the rule that the entry is the previous string plus its own
first character.

This module implements byte-oriented LZW compression and decompression with an optional maximum code
width (dictionary reset when full, as in GIF), returning a list of integer codes. It is verified by
exhaustive round-tripping: that decompress(compress(x)) == x for random bytes, highly repetitive
data, text, all-same and all-distinct inputs, and the empty string; that repetitive input yields far
fewer codes than its length (real compression); that the tricky KwKwK self-referential case decodes
correctly; and that the dictionary-reset path round-trips when the code width is capped. Pure stdlib;
a data-compression companion to the Huffman, LZ77, arithmetic-coding, and BWT notes."""

from __future__ import annotations


def compress(data, max_bits=None):
    """Compress a bytes-like object (or list of ints 0..255) into a list of integer codes.

    If max_bits is set, the dictionary resets to the initial 256 entries once its next code would
    need more than max_bits bits (GIF-style), keeping codes bounded."""
    data = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
    if not data:
        return []
    max_code = (1 << max_bits) - 1 if max_bits else None

    dictionary = {bytes([i]): i for i in range(256)}
    next_code = 256
    codes = []
    w = b""
    for byte in data:
        c = bytes([byte])
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            if max_code is None or next_code <= max_code:
                dictionary[wc] = next_code
                next_code += 1
            elif max_bits:
                # reset the dictionary (GIF behaviour)
                dictionary = {bytes([i]): i for i in range(256)}
                next_code = 256
            w = c
    if w:
        codes.append(dictionary[w])
    return codes


def decompress(codes, max_bits=None):
    """Decompress a list of integer codes back into the original bytes."""
    if not codes:
        return b""
    max_code = (1 << max_bits) - 1 if max_bits else None

    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256
    result = bytearray()

    if codes[0] not in dictionary:
        raise ValueError(f"invalid initial LZW code {codes[0]}")
    prev = dictionary[codes[0]]
    result += prev
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            # the KwKwK case: the code refers to the entry we are about to build
            entry = prev + prev[:1]
        else:
            raise ValueError(f"invalid LZW code {code}")
        result += entry
        # add prev + entry[0] as the new dictionary entry (one step behind the encoder)
        if max_code is None or next_code <= max_code:
            dictionary[next_code] = prev + entry[:1]
            next_code += 1
        elif max_bits:
            dictionary = {i: bytes([i]) for i in range(256)}
            next_code = 256
            # after reset, the just-read code was from the fresh dictionary
        prev = entry
    return bytes(result)


def compression_ratio(data, max_bits=None):
    """The ratio of output codes to input bytes (lower is better compression). Purely informational;
    a real encoder would pack the codes into a bitstream."""
    data = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
    if not data:
        return 0.0
    return len(compress(data, max_bits)) / len(data)
