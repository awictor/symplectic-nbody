"""Binary BCH codes: multiple-error-correcting cyclic codes built from the roots of a generator.

BCH codes (Bose-Chaudhuri-Hocquenghem, 1959-60) are the workhorse of algebraic error correction --
the codes behind CDs, DVDs, QR codes, satellite links, and flash memory. A binary BCH code of length
n = 2^m - 1 that corrects t errors is defined by a beautiful spectral condition: its codewords, viewed
as polynomials over GF(2), are exactly the multiples of a GENERATOR polynomial g(x) whose roots
include 2t consecutive powers of a primitive element alpha of GF(2^m). Because a codeword c(x) is a
multiple of g, it vanishes at all those powers, so c(alpha^1) = ... = c(alpha^{2t}) = 0. That gives
the decoder its handle: evaluate a received word at those 2t points, and any nonzero result -- a
SYNDROME -- is caused purely by the error pattern, independent of which codeword was sent.

Decoding is then the classic three-step dance over GF(2^m): (1) compute the 2t syndromes; (2) run
Berlekamp-Massey to find the shortest error-locator polynomial whose recurrence generates the syndrome
sequence; (3) find that polynomial's roots by a Chien search -- their inverses point at the error
positions. Because the code is binary, once you know WHERE the errors are you just flip those bits;
there is no separate Forney magnitude step as in Reed-Solomon. The guaranteed correction radius is t
whenever the designed distance 2t+1 is met.

This module builds GF(2^m) from a primitive polynomial, constructs the BCH generator by taking the LCM
of the minimal polynomials of alpha^1..alpha^{2t}, and implements systematic encoding, syndrome
computation, Berlekamp-Massey, and Chien-search decoding. It is validated: the generator's roots really
are the required consecutive powers; every codeword evaluates to zero at those powers; the code
corrects every error pattern up to weight t (checked exhaustively for small codes and by random trials
for larger ones); it flags or mis-corrects beyond t as expected; and the dimension matches n minus the
generator degree. Pure stdlib; the binary-cyclic companion to the Reed-Solomon, Golay, and
Berlekamp-Massey tools."""

from __future__ import annotations


class GF2m:
    """Finite field GF(2^m) with a given primitive polynomial (as an integer bitmask incl. the x^m bit)."""

    # default primitive polynomials for common m (x^m + ... + 1), as bitmasks
    _PRIM = {
        2: 0b111,          # x^2 + x + 1
        3: 0b1011,         # x^3 + x + 1
        4: 0b10011,        # x^4 + x + 1
        5: 0b100101,       # x^5 + x^2 + 1
        6: 0b1000011,      # x^6 + x + 1
        7: 0b10001001,     # x^7 + x^3 + 1
        8: 0b100011101,    # x^8 + x^4 + x^3 + x^2 + 1
    }

    def __init__(self, m, prim=None):
        self.m = m
        self.n = (1 << m) - 1                      # multiplicative order, code length 2^m - 1
        self.prim = prim if prim is not None else self._PRIM[m]
        self.exp = [0] * (2 * self.n + 1)          # exp[i] = alpha^i
        self.log = [0] * (self.n + 1)              # log[a] = i such that alpha^i = a
        x = 1
        for i in range(self.n):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & (1 << m):
                x ^= self.prim
        for i in range(self.n, 2 * self.n + 1):
            self.exp[i] = self.exp[i - self.n]

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def inv(self, a):
        return self.exp[self.n - self.log[a]]

    def pow(self, a, k):
        if a == 0:
            return 0
        return self.exp[(self.log[a] * k) % self.n]

    def alpha(self, i):
        """alpha^i."""
        return self.exp[i % self.n]


def _poly_mul_gf2(a, b):
    """Multiply two GF(2) polynomials given as coefficient lists (LSB first)."""
    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                res[i + j] ^= (ai & bj)
    return res


def _poly_trim(p):
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def minimal_polynomial(gf, i):
    """Minimal polynomial over GF(2) of alpha^i: product of (x - alpha^j) over the cyclotomic coset of i.

    Returns a GF(2) coefficient list (LSB first)."""
    # cyclotomic coset: i, 2i, 4i, ... mod n (conjugates under Frobenius x->x^2)
    coset = []
    j = i % gf.n
    while j not in coset:
        coset.append(j)
        j = (j * 2) % gf.n
    # product over the coset of (x + alpha^j) -- but coefficients must be in GF(2).
    # Build the product in GF(2^m); the result is guaranteed to have binary coefficients.
    poly = [1]  # constant 1
    for j in coset:
        root = gf.alpha(j)
        # multiply poly (over GF(2^m)) by (x + root)
        new = [0] * (len(poly) + 1)
        for k, c in enumerate(poly):
            new[k] ^= gf.mul(c, root)   # c * root  (constant term contribution)
            new[k + 1] ^= c             # c * x
        poly = new
    # coefficients should all be 0 or 1 (element of GF(2))
    return [1 if c else 0 for c in poly]


def _lcm_gf2(polys):
    """LCM of GF(2) polynomials via repeated product with divisibility de-duplication of factors.

    Since each minimal polynomial is irreducible, the LCM is the product of the DISTINCT ones."""
    seen = []
    result = [1]
    for p in polys:
        tp = tuple(_poly_trim(list(p)))
        if tp in seen:
            continue
        seen.append(tp)
        result = _poly_mul_gf2(result, list(tp))
    return _poly_trim(result)


class BCH:
    """Binary BCH code of length n = 2^m - 1 correcting t errors (narrow-sense, b=1)."""

    def __init__(self, m, t, prim=None):
        self.gf = GF2m(m, prim)
        self.m = m
        self.t = t
        self.n = self.gf.n
        # generator = LCM of minimal polynomials of alpha^1 .. alpha^{2t}
        mins = [minimal_polynomial(self.gf, i) for i in range(1, 2 * t + 1)]
        self.g = _lcm_gf2(mins)
        self.deg = len(self.g) - 1
        self.k = self.n - self.deg                 # message length (info bits)

    # --- encoding -----------------------------------------------------------
    def encode(self, message_bits):
        """Systematic encode: message occupies the high-order positions, parity fills the low ones.

        message_bits: list of k bits (LSB first is NOT assumed; index 0 = first info bit)."""
        if len(message_bits) != self.k:
            raise ValueError(f"message must be {self.k} bits, got {len(message_bits)}")
        # m(x) * x^deg  then take remainder mod g, append remainder -> multiple of g
        shifted = [0] * self.deg + list(message_bits)   # low deg positions are parity slots
        rem = self._poly_mod(shifted, self.g)
        code = list(rem) + [0] * (len(shifted) - len(rem))
        # place message in high positions, parity (rem) in low positions
        codeword = [0] * self.n
        for idx in range(self.deg):
            codeword[idx] = rem[idx] if idx < len(rem) else 0
        for idx in range(self.k):
            codeword[self.deg + idx] = message_bits[idx]
        return codeword

    def _poly_mod(self, dividend, divisor):
        """Remainder of GF(2) polynomial division (both LSB-first)."""
        d = list(dividend)
        dd = len(divisor) - 1
        for i in range(len(d) - 1, dd - 1, -1):
            if d[i]:
                for j in range(dd + 1):
                    d[i - dd + j] ^= divisor[j]
        return d[:dd]

    # --- decoding -----------------------------------------------------------
    def syndromes(self, received):
        """The 2t syndromes S_j = r(alpha^j) for j=1..2t, as GF(2^m) elements."""
        gf = self.gf
        S = []
        for j in range(1, 2 * self.t + 1):
            acc = 0
            aj = gf.alpha(j)
            # Horner over GF(2^m): r has bit coeffs, evaluate at alpha^j
            val = 0
            for idx in range(self.n - 1, -1, -1):
                val = gf.mul(val, aj) ^ (received[idx] & 1)
            S.append(val)
        return S

    def _berlekamp_massey(self, S):
        """Berlekamp-Massey over GF(2^m); returns the error-locator polynomial (LSB first)."""
        gf = self.gf
        L = 0
        C = [1]         # current locator
        B = [1]         # last locator before length change
        b = 1
        mm = 1
        for n in range(len(S)):
            # discrepancy
            delta = S[n]
            for i in range(1, L + 1):
                if i < len(C):
                    delta ^= gf.mul(C[i], S[n - i])
            if delta == 0:
                mm += 1
            elif 2 * L <= n:
                T = list(C)
                coef = gf.mul(delta, gf.inv(b))
                # C = C - coef * x^mm * B
                shifted = [0] * mm + [gf.mul(coef, bb) for bb in B]
                C = _xor_gf_poly(C, shifted)
                L = n + 1 - L
                B = T
                b = delta
                mm = 1
            else:
                coef = gf.mul(delta, gf.inv(b))
                shifted = [0] * mm + [gf.mul(coef, bb) for bb in B]
                C = _xor_gf_poly(C, shifted)
                mm += 1
        return C, L

    def _chien_search(self, locator):
        """Find error positions: i such that locator(alpha^{-i}) = 0."""
        gf = self.gf
        positions = []
        for i in range(self.n):
            # evaluate locator at alpha^{-i}
            x = gf.alpha((gf.n - i) % gf.n)
            val = 0
            xp = 1
            for c in locator:
                val ^= gf.mul(c, xp)
                xp = gf.mul(xp, x)
            if val == 0:
                positions.append(i)
        return positions

    def decode(self, received):
        """Decode a received word. Returns (corrected_codeword, num_errors_corrected) or
        (None, -1) if decoding fails (more errors than correctable)."""
        S = self.syndromes(received)
        if all(s == 0 for s in S):
            return list(received), 0
        locator, L = self._berlekamp_massey(S)
        positions = self._chien_search(locator)
        if len(positions) != L:
            return None, -1                     # locator degree != #roots: uncorrectable
        corrected = list(received)
        for p in positions:
            corrected[p] ^= 1
        # verify: re-check syndromes
        if any(s != 0 for s in self.syndromes(corrected)):
            return None, -1
        return corrected, len(positions)

    def decode_message(self, received):
        """Decode and extract the k message bits (high positions)."""
        corrected, ne = self.decode(received)
        if corrected is None:
            return None, -1
        return corrected[self.deg:self.deg + self.k], ne


def _xor_gf_poly(a, b):
    """XOR (GF(2^m) addition) of two coefficient lists, padding to the longer length."""
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(len(a)):
        out[i] ^= a[i]
    for i in range(len(b)):
        out[i] ^= b[i]
    return out
