"""Hamming codes: correcting a bit error with a handful of parity checks.

Send data over a noisy channel and bits flip. A plain parity bit can DETECT a single error but
not fix it. Richard Hamming's 1950 codes CORRECT it: by placing parity bits at the power-of-two
positions (1, 2, 4, 8, ...) so that each data bit is covered by a unique combination of checks,
the pattern of failed checks -- the syndrome -- reads out, in binary, the exact position of the
flipped bit. Flip it back and the message is restored.

The classic Hamming(7,4) code carries 4 data bits in 7 transmitted bits and corrects any single
error. In general Hamming(2^m - 1, 2^m - 1 - m) uses m parity bits to protect 2^m - 1 - m data
bits: 3 parity bits guard 4 data (the (7,4) code), 4 guard 11 (the (15,11) code), and the
overhead shrinks as the block grows. Every Hamming code has minimum distance 3 -- any two
codewords differ in at least 3 bits -- which is exactly what lets it correct 1 error (or detect
2). Adding one overall parity bit gives SECDED (single-error-correct, double-error-detect),
the scheme in ECC computer memory.

This module encodes data into a Hamming codeword, computes the syndrome, decodes and corrects a
single-bit error, and reports the code's parameters and minimum distance. It verifies that every
possible single-bit error in every codeword is corrected. Pure stdlib (bit lists); the
error-correction companion to the Shannon-entropy note.
"""

from __future__ import annotations


def _parity_positions(m: int):
    """The 1-based power-of-two positions that hold parity bits: 1, 2, 4, ..., 2^(m-1)."""
    return [1 << i for i in range(m)]


def code_parameters(m: int):
    """(n, k) for the Hamming code with m parity bits: n = 2^m - 1 total bits, k = n - m data
    bits. m=3 -> (7, 4); m=4 -> (15, 11)."""
    if m < 2:
        raise ValueError("m must be >= 2")
    n = (1 << m) - 1
    return n, n - m


def encode(data, m: int = 3):
    """Encode a list of k data bits into an n-bit Hamming codeword (1-based positions; parity
    bits at powers of two). Returns the codeword as a list of n bits."""
    n, k = code_parameters(m)
    if len(data) != k:
        raise ValueError(f"need exactly {k} data bits for m={m}")
    parity_pos = set(_parity_positions(m))
    code = [0] * (n + 1)  # index 1..n; index 0 unused
    # place data bits into the non-parity positions, in order
    di = 0
    for pos in range(1, n + 1):
        if pos not in parity_pos:
            code[pos] = data[di]
            di += 1
    # each parity bit p covers positions whose (1-based) index has that bit set -> even parity
    for p in _parity_positions(m):
        s = 0
        for pos in range(1, n + 1):
            if pos != p and (pos & p) and code[pos]:
                s ^= 1
        code[p] = s
    return code[1:]  # drop the unused index-0 slot


def syndrome(codeword, m: int = 3) -> int:
    """The syndrome of a received codeword: XOR of the positions of all set bits, weighted by
    each parity check. Zero means no detected error; otherwise it is the 1-based position of the
    single flipped bit."""
    n, _ = code_parameters(m)
    if len(codeword) != n:
        raise ValueError(f"codeword must have {n} bits for m={m}")
    s = 0
    for i, bit in enumerate(codeword):
        if bit:
            s ^= (i + 1)  # convert 0-based index to 1-based position
    return s


def decode(codeword, m: int = 3):
    """Decode a received codeword, correcting a single-bit error if present. Returns
    (data_bits, error_position) where error_position is the 1-based flipped position (0 if
    none)."""
    n, k = code_parameters(m)
    cw = list(codeword)
    err = syndrome(cw, m)
    if err:
        cw[err - 1] ^= 1  # flip the erroneous bit back
    parity_pos = set(_parity_positions(m))
    data = [cw[pos - 1] for pos in range(1, n + 1) if pos not in parity_pos]
    return data, err


def minimum_distance(m: int = 3) -> int:
    """Minimum Hamming distance of the code. Every Hamming code has distance 3 (corrects 1
    error, detects 2)."""
    return 3


def hamming_distance(a, b) -> int:
    """Number of positions at which two equal-length bit lists differ."""
    if len(a) != len(b):
        raise ValueError("bit lists must be the same length")
    return sum(1 for x, y in zip(a, b) if x != y)


def code_rate(m: int = 3) -> float:
    """Information rate k/n: the fraction of transmitted bits that carry data."""
    n, k = code_parameters(m)
    return k / n


def encode_secded(data, m: int = 3):
    """SECDED codeword: the Hamming codeword plus one overall parity bit appended, giving
    single-error-correct / double-error-detect. Returns n+1 bits."""
    cw = encode(data, m)
    return cw + [sum(cw) & 1]


def decode_secded(codeword, m: int = 3):
    """Decode a SECDED codeword. Returns (data, status) where status is 'ok', 'corrected', or
    'double_error' (detected, uncorrectable)."""
    n, k = code_parameters(m)
    if len(codeword) != n + 1:
        raise ValueError(f"SECDED codeword must have {n + 1} bits for m={m}")
    body = list(codeword[:n])
    overall = codeword[n]
    syn = syndrome(body, m)
    overall_ok = (sum(body) & 1) == overall
    if syn == 0 and overall_ok:
        status = "ok"
    elif not overall_ok:
        # overall parity is wrong -> odd number of errors; the syndrome locates the single one
        if syn:
            body[syn - 1] ^= 1
        status = "corrected"
    else:
        # syndrome nonzero but overall parity correct -> even (double) error, cannot correct
        status = "double_error"
    parity_pos = set(_parity_positions(m))
    data = [body[pos - 1] for pos in range(1, n + 1) if pos not in parity_pos]
    return data, status
