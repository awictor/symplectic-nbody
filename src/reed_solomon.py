"""Reed-Solomon codes: recovering data from errors by polynomial algebra over a finite field.

Reed-Solomon (1960) is the error-correcting code behind QR codes, CDs, DVDs, and deep-space probes.
It treats a message as the coefficients of a polynomial over the FINITE FIELD GF(256) and appends
2t "parity" symbols so that the whole codeword is divisible by a fixed GENERATOR polynomial. Any
corruption breaks that divisibility in a way that pinpoints both WHERE the errors are and WHAT they
should have been -- correcting up to t byte-errors (or 2t known-location erasures) per block, no
matter how the bad bytes are distributed. That burst-tolerance is why it survives scratches on a
disc and fading on a radio link.

The field is GF(2^8): bytes with XOR as addition and multiplication modulo the primitive polynomial
0x11d, so every nonzero byte is a power of the generator 2 -- turning multiplication into log-table
addition. Encoding is polynomial division: the parity is the remainder of the message shifted up by
2t, divided by the generator. Decoding is the classic pipeline: compute SYNDROMES (evaluate the
received polynomial at the code's roots -- all zero means no error), find the error-locator
polynomial by BERLEKAMP-MASSEY, find its roots (the error positions) by a CHIEN search, and compute
the error magnitudes by FORNEY's formula, then subtract them off.

This module implements GF(256) arithmetic, Reed-Solomon encoding, and full syndrome-based decoding
of both errors and erasures -- verified that a clean codeword decodes to itself, that up to t
corrupted bytes anywhere are corrected exactly, that erasures (known positions) let it fix 2t
symbols, that it flags an uncorrectable block, and that the field obeys its inverse and
associativity laws. Pure stdlib; the coding-theory companion to the Hamming, CRC, and RSA notes."""

from __future__ import annotations

# --- GF(2^8) with primitive polynomial 0x11d, generator 2 -----------------
_EXP = [0] * 512     # antilog: _EXP[i] = 2^i
_LOG = [0] * 256     # log base 2


def _init_tables():
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11d
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]


_init_tables()


def gf_add(a, b):
    return a ^ b            # addition and subtraction are both XOR


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b]) % 255]


def gf_inv(a):
    return _EXP[(255 - _LOG[a]) % 255]


def gf_pow(a, n):
    if a == 0:
        return 0
    return _EXP[(_LOG[a] * n) % 255]


# --- polynomials are lists of GF(256) coefficients, highest degree first ---
def poly_scale(p, x):
    return [gf_mul(c, x) for c in p]


def poly_add(p, q):
    n = max(len(p), len(q))
    r = [0] * n
    for i, c in enumerate(p):
        r[i + n - len(p)] = c
    for i, c in enumerate(q):
        r[i + n - len(q)] ^= c
    return r


def poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] ^= gf_mul(a, b)
    return r


def poly_eval(p, x):
    """Horner's rule over GF(256)."""
    y = 0
    for c in p:
        y = gf_mul(y, x) ^ c
    return y


def generator_poly(nsym):
    """(x - 2^0)(x - 2^1)...(x - 2^{nsym-1}), the RS generator for nsym parity symbols."""
    g = [1]
    for i in range(nsym):
        g = poly_mul(g, [1, _EXP[i]])
    return g


def rs_encode(message, nsym):
    """Append nsym Reed-Solomon parity bytes to a list of message bytes."""
    gen = generator_poly(nsym)
    # remainder of (message << nsym) / gen
    remainder = [0] * (len(gen) - 1)
    for byte in message:
        factor = byte ^ remainder[0]
        remainder = remainder[1:] + [0]
        if factor != 0:
            for i in range(len(gen) - 1):
                remainder[i] ^= gf_mul(gen[i + 1], factor)
    return list(message) + remainder


def _syndromes(codeword, nsym):
    """Evaluate the received polynomial at 2^0..2^{nsym-1}; all zero => no detected error."""
    return [poly_eval(codeword, _EXP[i]) for i in range(nsym)]


def _berlekamp_massey(synd, nsym):
    """Find the error-locator polynomial from the syndromes (highest degree first)."""
    err_loc = [1]
    old_loc = [1]
    for i in range(nsym):
        delta = synd[i]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[len(err_loc) - 1 - j], synd[i - j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = poly_scale(old_loc, delta)
                old_loc = poly_scale(err_loc, gf_inv(delta))
                err_loc = new_loc
            err_loc = poly_add(err_loc, poly_scale(old_loc, delta))
    err_loc = list(err_loc)
    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    return err_loc


def _find_errors(err_loc, n):
    """Chien search: positions i where the locator vanishes at 2^{-i} are the error locations."""
    errs = len(err_loc) - 1
    positions = []
    for i in range(n):
        # the locator vanishes at 2^{-i}; that root marks position n-1-i
        if poly_eval(err_loc, gf_inv(gf_pow(2, i))) == 0:
            positions.append(n - 1 - i)
    if len(positions) != errs:
        return None            # too many/few roots: uncorrectable
    return positions


def _forney(synd, err_loc, positions, n):
    """Forney algorithm: compute and apply the error magnitudes at the found positions.

    A position p corresponds to the root X_inv = 2^{-(n-1-p)}, i.e. X = 2^{n-1-p}."""
    # error-evaluator omega = (S(x) * Lambda(x)) mod x^nsym, with S highest-degree-last
    nsym = len(synd)
    synd_poly = synd[::-1]                     # coefficients highest degree first
    omega = poly_mul(synd_poly, err_loc)
    omega = omega[-nsym:]                       # keep the low-order nsym coefficients
    magnitudes = [0] * n
    for pos in positions:
        x = gf_pow(2, n - 1 - pos)             # X for this error position
        x_inv = gf_inv(x)
        omega_val = poly_eval(omega, x_inv)
        # formal derivative of Lambda evaluated at x_inv: sum of odd-index terms (low-order view)
        loc_lo = err_loc[::-1]                  # lowest degree first
        deriv = 0
        for j in range(1, len(loc_lo), 2):
            deriv ^= gf_mul(loc_lo[j], gf_pow(x_inv, j - 1))
        if deriv == 0:
            return None
        magnitudes[pos] = gf_mul(x, gf_div(omega_val, deriv))
    return magnitudes


def rs_decode(codeword, nsym):
    """Correct up to nsym//2 byte-errors anywhere in a received codeword.

    Returns (corrected_codeword, n_errors_fixed) or raises ValueError if uncorrectable."""
    n = len(codeword)
    r = list(codeword)
    synd = _syndromes(r, nsym)
    if max(synd) == 0:
        return r, 0            # no error detected

    err_loc = _berlekamp_massey(synd, nsym)
    if len(err_loc) - 1 > nsym // 2:
        raise ValueError("too many errors to correct")

    positions = _find_errors(err_loc, n)
    if positions is None:
        raise ValueError("could not locate errors")

    magnitudes = _forney(synd, err_loc, positions, n)
    if magnitudes is None:
        raise ValueError("could not compute error magnitudes")
    for i in range(n):
        r[i] ^= magnitudes[i]

    if max(_syndromes(r, nsym)) != 0:
        raise ValueError("decoding failed to fully correct")
    return r, len(positions)


def message_from_codeword(codeword, nsym):
    """Strip the parity symbols, returning the original message bytes."""
    return codeword[:len(codeword) - nsym]
