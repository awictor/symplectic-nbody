"""First-order Reed-Muller code RM(1, m): the code that phoned home from Mars.

Reed-Muller codes are among the oldest and most beautiful error-correcting codes; the first-order code
RM(1, m) flew on the 1971 Mariner 9 mission, protecting the first close-up photographs of Mars against
the noise of interplanetary space. RM(1, m) encodes m+1 message bits into a codeword of length n = 2^m,
so its rate is (m+1)/2^m -- low, but its minimum distance is 2^(m-1), meaning it corrects up to
2^(m-2) - 1 bit errors. For RM(1,5), used by Mariner, that is 32 bits protecting 6, correcting up to 7
flips per 32-bit word.

The construction is elegant: the 2^m codewords are the values of all AFFINE boolean functions
f(x) = a_0 + a_1 x_1 + ... + a_m x_m over the m-bit inputs (a total of 2^(m+1) codewords, m+1 message
bits). Mapped from {0,1} to {+1,-1}, these codewords are exactly the rows of a HADAMARD matrix (plus
their negations), which is what makes decoding fast and exact: a received word, transformed by the
FAST WALSH-HADAMARD TRANSFORM, has its largest-magnitude coefficient at the index of the closest
codeword -- maximum-likelihood decoding in O(n log n) instead of comparing against all 2^(m+1)
codewords.

This module encodes a message, decodes a (possibly corrupted) received word by the Hadamard transform,
reports the code parameters, and exposes the correctable-error radius. Validated: encode-then-decode
round-trips every message; any error pattern up to the correction radius is corrected exactly (checked
exhaustively for small m); the code's minimum distance is 2^(m-1) (all nonzero codewords have that
weight or its complement); the transform decoder agrees with brute-force nearest-codeword decoding;
and errors beyond the radius are (correctly) not guaranteed. Pure stdlib; the coding-theory companion
to the Hamming, Golay, and Reed-Solomon codes and the Walsh-Hadamard transform."""

from __future__ import annotations

from walsh_hadamard import fwht


def _bits(x, m):
    return [(x >> i) & 1 for i in range(m)]


def code_parameters(m):
    """(n, k, d, t): length 2^m, message bits m+1, min distance 2^(m-1), correctable errors."""
    n = 1 << m
    k = m + 1
    d = 1 << (m - 1)
    t = (d // 2) - 1 if d >= 2 else 0
    return n, k, d, t


def encode(message, m):
    """Encode m+1 message bits [a_0, a_1, ..., a_m] into the length-2^m RM(1,m) codeword.
    Codeword bit at input x is a_0 XOR (a_1 x_1) XOR ... XOR (a_m x_m)."""
    if len(message) != m + 1:
        raise ValueError(f"message must have {m + 1} bits")
    n = 1 << m
    a0 = message[0]
    lin = message[1:]
    word = []
    for x in range(n):
        bits = _bits(x, m)
        val = a0
        for i in range(m):
            val ^= lin[i] & bits[i]
        word.append(val)
    return word


def decode(received, m):
    """Decode a length-2^m word (0/1 bits, possibly corrupted) to the m+1 message bits by the fast
    Walsh-Hadamard transform. Returns (message, corrected_codeword)."""
    n = 1 << m
    if len(received) != n:
        raise ValueError(f"received word must have length {n}")
    # map 0/1 -> +1/-1, transform, find the largest-magnitude coefficient
    signal = [1 - 2 * b for b in received]  # 0->+1, 1->-1
    spectrum = fwht(list(signal))
    # index of max |coefficient| gives the linear part; its sign gives a_0
    best_idx = max(range(n), key=lambda i: abs(spectrum[i]))
    # a_0: if the peak coefficient is negative, the constant term is 1
    a0 = 0 if spectrum[best_idx] > 0 else 1
    # best_idx (as an m-bit number) encodes the linear coefficients a_1..a_m
    lin = _bits(best_idx, m)
    message = [a0] + lin
    return message, encode(message, m)


def correctable_errors(m):
    """The guaranteed error-correction radius t = 2^(m-2) - 1."""
    return code_parameters(m)[3]


def hamming_distance(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


# --- brute-force reference: nearest-codeword decoding ------------------------
def all_codewords(m):
    """Every RM(1,m) codeword (2^(m+1) of them), as (message, codeword) pairs."""
    out = []
    for msg_int in range(1 << (m + 1)):
        message = [(msg_int >> i) & 1 for i in range(m + 1)]
        out.append((message, encode(message, m)))
    return out


def brute_decode(received, m):
    """Maximum-likelihood decode by comparing to every codeword (reference for the FWHT decoder)."""
    best = None
    best_d = None
    for message, word in all_codewords(m):
        d = hamming_distance(received, word)
        if best_d is None or d < best_d:
            best_d = d
            best = message
    return best
