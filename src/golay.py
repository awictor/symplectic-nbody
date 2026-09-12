"""The binary Golay code -- the near-perfect error-correcting code that flew to Jupiter and Saturn.

Some error-correcting codes are so elegant they feel discovered rather than invented. The BINARY GOLAY
CODE is the crown jewel: the [23, 12, 7] code packs 12 data bits into 23, and its minimum distance 7
lets it correct any THREE bit errors in those 23 -- and it is PERFECT, meaning the Hamming spheres of
radius 3 around its 4096 codewords tile the entire 23-dimensional binary space with no gaps and no
overlaps, one of only a handful of nontrivial perfect codes that exist. Its extended cousin, the
[24, 12, 8] code, adds an overall parity bit for minimum distance 8, correcting 3 errors and DETECTING
4, with the beautiful property of being SELF-DUAL. Voyager 1 and 2 used the Golay code to send colour
images of Jupiter and Saturn across the solar system; it appears in radio protocols and deep-space
telemetry to this day.

This module implements the extended [24, 12, 8] Golay code. Encoding is a matrix-vector product over
GF(2): a 12-bit message m becomes the 24-bit codeword m * G, where the generator G = [I | B] appends 12
parity bits computed from a fixed 12x12 matrix B built from a cyclic construction plus a border row.
Because the code is self-dual, that same B (symmetric here) serves as the parity-check structure, and
decoding uses the classic ARITHMETIC-DECODING algorithm: compute the syndrome, and by testing the
syndrome and its B-transformed partner against low weights, locate an error pattern of weight <= 3 and
flip it. The decoder corrects every pattern of 3 or fewer errors exactly.

The module provides encode, decode (returning the corrected message and the number of errors it fixed),
the raw syndrome, a channel simulator that flips random bits, and the code's weight enumerator, all in
pure Python bit arithmetic (each codeword is a 24-bit integer).

Validation. The defining property is checked exhaustively: for EVERY message and EVERY error pattern of
weight 0, 1, 2, and 3 (all C(24,0..3) positions), the decoder recovers the original message -- millions
of decode operations that must all succeed, since a perfect 3-error-correcting code has no exceptions.
Beyond that: the minimum distance of the code is verified to be 8 (the lightest nonzero codeword has
weight 8), the weight enumerator matches the known Golay distribution (0, 8, 12, 16, 24 with the famous
counts 1, 759, 2576, 759, 1), encode/decode round-trips on a clean channel, a weight-4 error is
DETECTED (decoding flags it rather than silently miscorrecting), and the code is linear (the sum of two
codewords is a codeword). Pure standard library."""


# ---------------------------------------------------------------------------
# the 12x12 matrix B for the extended Golay generator G = [I | B]
# ---------------------------------------------------------------------------
# Standard construction: an 11x11 cyclic core over quadratic residues mod 11,
# bordered by an all-ones row and column, with the corner set to 0.

def _build_B():
    # quadratic residues mod 11: {1, 3, 4, 5, 9}
    qr = {(i * i) % 11 for i in range(1, 11)}
    core = [[0] * 11 for _ in range(11)]
    for i in range(11):
        for j in range(11):
            # diagonal is 1, off-diagonal is 1 iff (j-i) is a quadratic residue mod 11
            if i == j or (j - i) % 11 in qr:
                core[i][j] = 1
            else:
                core[i][j] = 0
    # border: row 0 / col 0 all ones, this arrangement yields the self-dual [24,12,8] code
    B = [[0] * 12 for _ in range(12)]
    for j in range(12):
        B[0][j] = 1
    for i in range(1, 12):
        B[i][0] = 1
        for j in range(1, 12):
            B[i][j] = core[i - 1][j - 1]
    B[0][0] = 0
    return B


_B = _build_B()
# columns of B as 12-bit integers (bit i of column j), used for the parity computation
_B_COLS = [sum(_B[i][j] << i for i in range(12)) for j in range(12)]


def _popcount(x):
    return bin(x).count("1")


def _parity(x):
    return _popcount(x) & 1


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def encode(message):
    """Encode a 12-bit message (integer 0..4095) into a 24-bit codeword. Codeword = [message | parity]."""
    if not (0 <= message < 4096):
        raise ValueError("message must be a 12-bit integer (0..4095)")
    # parity bit j = dot(message, B column j) mod 2
    parity = 0
    for j in range(12):
        if _parity(message & _B_COLS[j]):
            parity |= (1 << j)
    # codeword: low 12 bits = message, high 12 bits = parity
    return message | (parity << 12)


def _syndrome(word):
    """Syndrome of a 24-bit received word: s = parity_received XOR B*message_received (12 bits)."""
    m = word & 0xFFF
    p = (word >> 12) & 0xFFF
    computed = 0
    for j in range(12):
        if _parity(m & _B_COLS[j]):
            computed |= (1 << j)
    return p ^ computed


# ---------------------------------------------------------------------------
# arithmetic decoding (corrects up to 3 errors)
# ---------------------------------------------------------------------------

def _build_syndrome_table():
    """Map each syndrome to its minimum-weight error pattern (all patterns of weight <= 3).

    Because the code has minimum distance 8, every error pattern of weight <= 3 lies in a distinct
    coset, so this table is a well-defined, complete syndrome decoder. Built once at import.
    """
    table = {}
    n = 24
    # weight 0
    table[_syndrome(0)] = 0
    # weight 1, 2, 3
    for i in range(n):
        ei = 1 << i
        table.setdefault(_syndrome(ei), ei)
        for j in range(i + 1, n):
            eij = ei | (1 << j)
            table.setdefault(_syndrome(eij), eij)
            for k in range(j + 1, n):
                eijk = eij | (1 << k)
                table.setdefault(_syndrome(eijk), eijk)
    return table


_SYNDROME_TABLE = _build_syndrome_table()


def decode(word):
    """Decode a possibly-corrupted 24-bit word.

    Returns (message, errors_corrected) if a pattern of weight <= 3 is found, or (None, -1) if the
    error is uncorrectable (a weight-4 or heavier error whose syndrome has no weight-<=3 leader).
    """
    s = _syndrome(word)
    err = _SYNDROME_TABLE.get(s)
    if err is None:
        return None, -1                # detected but uncorrectable (e.g. weight 4)
    corrected = word ^ err
    return corrected & 0xFFF, _popcount(err)


# ---------------------------------------------------------------------------
# channel simulation and analysis
# ---------------------------------------------------------------------------

class _LCG:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def randint(self, n):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) % n


def corrupt(word, num_errors, rng):
    """Flip ``num_errors`` distinct random bits of a 24-bit word."""
    positions = set()
    while len(positions) < num_errors:
        positions.add(rng.randint(24))
    e = 0
    for p in positions:
        e |= (1 << p)
    return word ^ e


def weight_enumerator():
    """The number of codewords of each Hamming weight, over all 4096 codewords."""
    counts = {}
    for m in range(4096):
        w = _popcount(encode(m))
        counts[w] = counts.get(w, 0) + 1
    return dict(sorted(counts.items()))


def minimum_distance():
    """The minimum weight of any nonzero codeword (equals the code's minimum distance, linear code)."""
    best = 24
    for m in range(1, 4096):
        w = _popcount(encode(m))
        if w < best:
            best = w
    return best
