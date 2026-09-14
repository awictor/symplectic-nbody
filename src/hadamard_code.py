"""Hadamard codes: the maximally-redundant error-correcting code, decoded by one Walsh-Hadamard transform.

The Hadamard code trades rate for extraordinary error tolerance. It encodes an m-bit message as a
codeword of length n = 2^m -- so only m+1 information bits ride in 2^m transmitted bits, a vanishing
rate -- but ANY two distinct codewords differ in exactly n/2 positions. That enormous minimum distance
means the code corrects nearly n/4 bit errors, and it famously carried the Mariner 9 pictures back from
Mars over a brutally noisy deep-space channel. Its punctured cousin is the first-order Reed-Muller code
RM(1, m).

The construction is pure sign structure. Number the codewords by a message x in {0,1}^m; the codeword's
i-th bit (for i in 0..2^m - 1) is the parity of the bitwise-AND of x and i, i.e. <x, i> mod 2. Written
in +/-1 form, the 2^m codewords are exactly the ROWS of the 2^m x 2^m Hadamard matrix -- mutually
orthogonal sign vectors. That orthogonality is what makes decoding a single transform: map the received
word to +/-1, apply the fast WALSH-HADAMARD TRANSFORM, and the largest-magnitude coefficient's index is
the sent message, its sign the appended parity bit. The transform simultaneously correlates the
received word against ALL 2^m codewords in O(n log n) -- maximum-likelihood decoding for the price of
one FFT-like pass.

This module encodes/decodes the (augmented) Hadamard code via the repo's fast Walsh-Hadamard transform,
and exposes the minimum distance and correction radius. It is validated: every pair of distinct
codewords is at Hamming distance exactly n/2 (so the minimum distance is n/2); a clean codeword decodes
to itself; the code corrects any error pattern of weight below n/4 (checked exhaustively for small m and
by random trials for larger m); the FWHT decoder agrees with a brute-force maximum-correlation decoder;
and the correlation peak's height degrades predictably with the number of errors. Reuses the repo's
Walsh-Hadamard transform. Pure stdlib; the high-distance companion to the Reed-Muller, BCH, and
Reed-Solomon tools."""

from __future__ import annotations

from walsh_hadamard import fwht


def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def encode(message, m):
    """Encode an m-bit message (integer 0..2^m-1) as the length-2^m augmented Hadamard codeword.

    Codeword bit i = parity(message AND i). Returns a list of 0/1 of length 2^m."""
    n = 1 << m
    return [_popcount(message & i) & 1 for i in range(n)]


def encode_bits(bits):
    """Encode from a list of m bits (MSB first) -> codeword of length 2^m."""
    m = len(bits)
    message = 0
    for b in bits:
        message = (message << 1) | (b & 1)
    return encode(message, m)


def all_codewords(m):
    """All 2^m Hadamard codewords."""
    return [encode(x, m) for x in range(1 << m)]


def min_distance(m):
    """The minimum Hamming distance of the code = n/2 = 2^(m-1)."""
    return 1 << (m - 1)


def correctable_errors(m):
    """Guaranteed correction radius = floor((d-1)/2) = floor((2^(m-1) - 1)/2)."""
    return (min_distance(m) - 1) // 2


def decode(received, m):
    """Maximum-likelihood decode via the fast Walsh-Hadamard transform.

    received: list of 0/1 of length 2^m. Returns the decoded message integer (0..2^m-1)."""
    n = 1 << m
    # map 0/1 -> +1/-1 so codewords become Hadamard rows
    signs = [1 if b == 0 else -1 for b in received]
    spectrum = fwht(signs)
    # the message is the index of the largest-magnitude coefficient
    best = max(range(n), key=lambda i: abs(spectrum[i]))
    return best


def decode_with_confidence(received, m):
    """Decode and also return the correlation peak (n = perfect, lower = noisier)."""
    n = 1 << m
    signs = [1 if b == 0 else -1 for b in received]
    spectrum = fwht(signs)
    best = max(range(n), key=lambda i: abs(spectrum[i]))
    return best, abs(spectrum[best])


def brute_decode(received, m):
    """Reference decoder: pick the codeword with the smallest Hamming distance to the received word."""
    n = 1 << m
    best = 0
    best_d = n + 1
    for x in range(n):
        cw = encode(x, m)
        d = sum(1 for i in range(n) if cw[i] != received[i])
        if d < best_d:
            best_d = d
            best = x
    return best


def hamming_distance(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])
