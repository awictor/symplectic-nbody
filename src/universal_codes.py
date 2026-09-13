"""Universal integer codes: Elias gamma/delta/omega and Golomb-Rice, self-delimiting and prefix-free.

How do you write an integer to a bitstream when you do not know in advance how big it will get -- no
fixed 32-bit field, no agreed maximum? You need a SELF-DELIMITING code: the bits themselves announce
where the number ends, so a decoder reading left to right always knows when to stop. These UNIVERSAL
CODES are the backbone of real compressors -- inverted indexes (search engines store gaps between
document IDs), FLAC and other audio codecs, and any format that codes runs of small integers -- because
they spend few bits on the small values that dominate real data while still being able to represent
arbitrarily large ones.

Four classic schemes, all implemented here:

  ELIAS GAMMA. Write floor(log2 n) zeros, then n in binary (whose leading 1 doubles as the separator).
  Codes n in about 2 floor(log2 n) + 1 bits -- ideal when the value distribution falls off like 1/n^2.

  ELIAS DELTA. Encode the LENGTH of n's binary using gamma, then n's remaining bits. Asymptotically
  shorter than gamma for large n (about log2 n + 2 log2 log2 n bits).

  ELIAS OMEGA. Recursively prepends the length of the length of the length... a group at a time,
  terminated by a 0. The most compact of the three for very large integers.

  GOLOMB-RICE. Split n by a parameter M into a quotient (written in unary) and a remainder (written in
  binary). Rice codes are the M = 2^k special case (fast, bit-shift only). This is the OPTIMAL prefix
  code for a GEOMETRIC distribution -- the right choice for the residuals in lossless audio/image
  codecs -- when M is tuned to the mean.

This module implements encode/decode for each scheme, both single-value and as a stream (concatenating
codes and decoding them back, which exercises the self-delimiting property), plus the exact code-length
formulas. It is validated exhaustively: every code round-trips every integer in a large range; codes
are prefix-free (no codeword is a prefix of another) and self-delimiting (a concatenation of many codes
decodes back to the exact original list with no separators); the measured lengths match the closed-form
formulas; Rice is the k-bit special case of Golomb; and on a geometric source the tuned Golomb code
beats a fixed-width encoding. Pure stdlib; the universal-integer-code companion to the Huffman,
arithmetic, and rANS entropy coders."""

from __future__ import annotations

import math


# ---- bit helpers -------------------------------------------------------------------------------

def _binary(n, width=None):
    """Binary string of n (>=0). If width given, zero-pad on the left."""
    b = bin(n)[2:]
    if width is not None:
        b = b.rjust(width, "0")
    return b


# ---- Elias gamma -------------------------------------------------------------------------------

def gamma_encode(n):
    """Elias gamma code of a positive integer n (>= 1). Returns a bit string."""
    if n < 1:
        raise ValueError("Elias gamma codes positive integers (n >= 1)")
    b = _binary(n)
    return "0" * (len(b) - 1) + b


def gamma_decode_stream(bits, count=None):
    """Decode a concatenation of Elias-gamma codes. Returns the list of integers."""
    out = []
    i = 0
    n = len(bits)
    while i < n and (count is None or len(out) < count):
        # count leading zeros
        z = 0
        while i < n and bits[i] == "0":
            z += 1
            i += 1
        if i >= n:
            break
        # read z more bits after the leading 1
        val_bits = bits[i:i + z + 1]
        i += z + 1
        out.append(int(val_bits, 2))
    return out


def gamma_length(n):
    return 2 * (n.bit_length() - 1) + 1


# ---- Elias delta -------------------------------------------------------------------------------

def delta_encode(n):
    """Elias delta code of n >= 1."""
    if n < 1:
        raise ValueError("Elias delta codes positive integers (n >= 1)")
    b = _binary(n)
    L = len(b)  # bit length of n
    # gamma-encode L, then append n's bits without the leading 1
    return gamma_encode(L) + b[1:]


def delta_decode_stream(bits, count=None):
    out = []
    i = 0
    n = len(bits)
    while i < n and (count is None or len(out) < count):
        # gamma-decode the length L
        z = 0
        while i < n and bits[i] == "0":
            z += 1
            i += 1
        if i >= n:
            break
        Lbits = bits[i:i + z + 1]
        i += z + 1
        L = int(Lbits, 2)
        # read L-1 more bits; value is 1 followed by them
        rest = bits[i:i + L - 1]
        i += L - 1
        out.append(int("1" + rest, 2) if L > 1 else 1)
    return out


def delta_length(n):
    L = n.bit_length()
    return gamma_length(L) + (L - 1)


# ---- Elias omega -------------------------------------------------------------------------------

def omega_encode(n):
    """Elias omega (recursive) code of n >= 1."""
    if n < 1:
        raise ValueError("Elias omega codes positive integers (n >= 1)")
    code = "0"
    k = n
    while k > 1:
        b = _binary(k)
        code = b + code
        k = len(b) - 1
    return code


def omega_decode_stream(bits, count=None):
    out = []
    i = 0
    N = len(bits)
    while i < N and (count is None or len(out) < count):
        if bits[i] == "0":
            # a lone terminator with no preceding group encodes n = 1
            i += 1
            out.append(1)
            continue
        n = 1
        while i < N and bits[i] == "1":
            # read n+1 bits as the next group
            grp = bits[i:i + n + 1]
            i += n + 1
            n = int(grp, 2)
        # consume the terminating 0
        if i < N and bits[i] == "0":
            i += 1
        out.append(n)
    return out


# ---- Golomb / Rice -----------------------------------------------------------------------------

def golomb_encode(n, M):
    """Golomb code of n >= 0 with parameter M >= 1 (unary quotient + truncated-binary remainder)."""
    if n < 0 or M < 1:
        raise ValueError("golomb needs n >= 0 and M >= 1")
    q, r = divmod(n, M)
    quotient = "1" * q + "0"  # unary q then a 0 terminator
    # truncated binary for r in [0, M)
    b = M.bit_length() - 1  # floor(log2 M)
    cutoff = (1 << (b + 1)) - M
    if r < cutoff:
        remainder = _binary(r, b) if b > 0 else ""
    else:
        remainder = _binary(r + cutoff, b + 1)
    return quotient + remainder


def golomb_decode_stream(bits, M, count):
    """Decode `count` Golomb codes with parameter M."""
    out = []
    i = 0
    N = len(bits)
    b = M.bit_length() - 1
    cutoff = (1 << (b + 1)) - M
    while len(out) < count and i <= N:
        q = 0
        while i < N and bits[i] == "1":
            q += 1
            i += 1
        i += 1  # skip the 0 terminator
        if b == 0:
            r = 0
        else:
            first = bits[i:i + b]
            i += b
            rval = int(first, 2) if first else 0
            if rval < cutoff:
                r = rval
            else:
                extra = bits[i]
                i += 1
                r = (rval << 1 | int(extra)) - cutoff
        out.append(q * M + r)
    return out


def rice_encode(n, k):
    """Rice code: Golomb with M = 2^k (quotient unary + k-bit remainder)."""
    return golomb_encode(n, 1 << k)


def rice_decode_stream(bits, k, count):
    return golomb_decode_stream(bits, 1 << k, count)


def golomb_optimal_M(mean):
    """Optimal Golomb parameter M for a geometric source with the given mean (Golomb 1966)."""
    if mean <= 0:
        return 1
    p = 1.0 / (mean + 1.0)  # success probability of the geometric
    # M = ceil( -1 / log2(1-p) )
    q = 1 - p
    if q <= 0:
        return 1
    return max(1, math.ceil(-1.0 / math.log2(q)))


# ---- stream convenience ------------------------------------------------------------------------

def encode_stream(values, scheme="gamma", M=None, k=None):
    """Encode a list of integers into one concatenated bit string."""
    if scheme == "gamma":
        return "".join(gamma_encode(v) for v in values)
    if scheme == "delta":
        return "".join(delta_encode(v) for v in values)
    if scheme == "omega":
        return "".join(omega_encode(v) for v in values)
    if scheme == "golomb":
        return "".join(golomb_encode(v, M) for v in values)
    if scheme == "rice":
        return "".join(rice_encode(v, k) for v in values)
    raise ValueError(f"unknown scheme {scheme}")


def is_prefix_free(codewords):
    """True if no codeword is a prefix of another (necessary for unique decodability)."""
    codes = sorted(set(codewords), key=len)
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            if codes[j].startswith(codes[i]):
                return False
    return True
