"""Cyclic redundancy check: catching transmission errors with polynomial division.

A CRC is the checksum stamped on almost every frame of digital data -- Ethernet packets, ZIP
files, PNG chunks, disk sectors. The idea: treat the message as the coefficients of a big
polynomial over GF(2) (arithmetic mod 2, where addition is XOR), divide by a fixed "generator"
polynomial, and append the remainder. The receiver divides the whole thing again; a zero
remainder means no detected error, and any corruption that is not an exact multiple of the
generator shows up as a nonzero remainder.

Why it works so well: a well-chosen generator of degree r guarantees detection of

  * any single-bit error,
  * any odd number of bit errors (if the generator has an even number of terms),
  * any burst error shorter than r+1 bits,

and misses a random error only with probability about 2^-r. That is a lot of protection for r
check bits (32 for CRC-32), computed with nothing but shifts and XORs -- which is why CRCs are
cheap enough to run in hardware at line rate.

This module computes CRCs by explicit bit-at-a-time polynomial division (the textbook algorithm,
so the mechanism is visible), supports the standard CRC-8, CRC-16-CCITT, and CRC-32 parameters,
verifies transmitted frames, and demonstrates the error-detection guarantees. It reproduces the
published check value of CRC-32 over the ASCII string "123456789". Pure stdlib; the
error-detection companion to the Hamming-code note (Hamming corrects, CRC detects).
"""

from __future__ import annotations

# (width, polynomial, init, reflect_in/out, xor_out) for the standard CRCs.
PARAMS = {
    "CRC-8": dict(width=8, poly=0x07, init=0x00, refin=False, refout=False, xorout=0x00),
    "CRC-16-CCITT": dict(width=16, poly=0x1021, init=0xFFFF, refin=False, refout=False, xorout=0x0000),
    "CRC-32": dict(width=32, poly=0x04C11DB7, init=0xFFFFFFFF, refin=True, refout=True, xorout=0xFFFFFFFF),
}


def _reflect(value: int, width: int) -> int:
    """Reverse the low `width` bits of value (bit-order reflection used by CRC-32)."""
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def crc(data: bytes, width: int, poly: int, init: int = 0, refin: bool = False,
        refout: bool = False, xorout: int = 0) -> int:
    """Compute a CRC by bit-at-a-time polynomial division over GF(2).

    width  : degree of the generator (number of check bits)
    poly   : generator polynomial (without the implicit leading x^width term)
    init   : initial register value
    refin  : reflect each input byte (LSB-first) before processing
    refout : reflect the final register
    xorout : final XOR mask
    """
    topbit = 1 << (width - 1)
    mask = (1 << width) - 1
    reg = init
    for byte in data:
        if refin:
            byte = _reflect(byte, 8)
        reg ^= (byte << (width - 8)) & mask if width >= 8 else byte
        for _ in range(8):
            if reg & topbit:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    if refout:
        reg = _reflect(reg, width)
    return reg ^ xorout


def crc_named(data: bytes, name: str = "CRC-32") -> int:
    """Compute a CRC using one of the standard parameter sets in PARAMS."""
    if name not in PARAMS:
        raise ValueError(f"unknown CRC: {name}; choose from {list(PARAMS)}")
    return crc(data, **PARAMS[name])


def append_crc(data: bytes, name: str = "CRC-32") -> bytes:
    """Return data with its CRC appended (big-endian), as it would go on the wire."""
    p = PARAMS[name]
    value = crc_named(data, name)
    nbytes = p["width"] // 8
    return data + value.to_bytes(nbytes, "big")


def check_frame(frame: bytes, name: str = "CRC-32") -> bool:
    """Verify a frame (data + appended CRC): recompute the CRC of the data portion and compare
    to the trailing checksum. True if they match (no error detected)."""
    p = PARAMS[name]
    nbytes = p["width"] // 8
    if len(frame) < nbytes:
        return False
    data, tail = frame[:-nbytes], frame[-nbytes:]
    return crc_named(data, name) == int.from_bytes(tail, "big")


def detects_error(data: bytes, corrupt: bytes, name: str = "CRC-32") -> bool:
    """True if the CRC differs between the original and a corrupted message (i.e. the error is
    detected). Equal-length inputs assumed."""
    return crc_named(data, name) != crc_named(corrupt, name)


def flip_bit(data: bytes, bit_index: int) -> bytes:
    """Return a copy of data with a single bit flipped (bit_index counts from the MSB of byte
    0). Handy for exercising the single-error-detection guarantee."""
    b = bytearray(data)
    byte_i, bit_in_byte = divmod(bit_index, 8)
    b[byte_i] ^= 1 << (7 - bit_in_byte)
    return bytes(b)
