"""SHA-256 from scratch -- the hash function under TLS certificates, Bitcoin, and git commits.

A cryptographic hash maps a message of any length to a fixed 256-bit digest such that you cannot find
two messages with the same digest, cannot recover the message from the digest, and any one-bit change
scrambles the output completely. SHA-256 (NIST FIPS 180-4, 2001) is the workhorse: it certifies every
HTTPS connection, names every git object and Bitcoin block, and underlies HMAC message authentication.
Implementing it from the specification -- not by calling a library -- is a rite of passage that reveals
exactly how the avalanche effect is engineered from simple bitwise operations.

The construction is Merkle-Damgard: chop the padded message into 512-bit blocks and iterate a
COMPRESSION FUNCTION that folds each block into a running 256-bit state (eight 32-bit words). Three
ingredients do all the work, all on 32-bit words modulo 2^32:

  * PADDING. Append a single 1 bit, then zeros, then the 64-bit message length, so the total is a
    multiple of 512 bits. This makes the length part of the input (defeating trivial extension of
    same-length messages) and guarantees full blocks.

  * MESSAGE SCHEDULE. Each 512-bit block gives sixteen 32-bit words; these are stretched to sixty-four
    by a recurrence mixing earlier words through the small-sigma rotations, so every output word depends
    on the whole block.

  * ROUND FUNCTION. Sixty-four rounds, each stirring the eight state words with the choice (Ch) and
    majority (Maj) functions, the big-sigma rotations, a round constant (the fractional parts of the
    cube roots of the first 64 primes), and one schedule word. After the rounds, the block's result is
    added back to the incoming state -- the Davies-Meyer feed-forward that makes the function one-way.

This module implements the full algorithm: rotation and logical primitives, padding, the message
schedule, the compression function, and a streaming API (update / digest) plus one-shot helpers that
return the digest as bytes or hex. It also builds HMAC-SHA256 on top, the standard keyed message
authentication code. Pure standard library -- integer arithmetic only, no ``hashlib`` in the
implementation.

Validation. The digest is checked BIT-FOR-BIT against Python's ``hashlib.sha256`` -- the reference used
only in the tests -- on the empty string, the classic "abc", multi-block inputs, every length from 0 to
200 bytes (exercising all padding cases including the awkward 55/56/64-byte boundaries), and hundreds of
random byte strings; they must match on every single one. The published NIST test vectors are matched by
hand. The streaming API is checked to equal the one-shot digest regardless of how the input is chopped.
HMAC-SHA256 is validated against ``hmac`` on random key/message pairs and against the RFC 4231 vectors."""


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_MASK = 0xFFFFFFFF

# initial hash values: fractional parts of the square roots of the first 8 primes
_H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

# round constants: fractional parts of the cube roots of the first 64 primes
_K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]


# ---------------------------------------------------------------------------
# 32-bit primitives
# ---------------------------------------------------------------------------

def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & _MASK


def _shr(x, n):
    return x >> n


def _ch(x, y, z):
    return (x & y) ^ (~x & z)


def _maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)


def _big_sigma0(x):
    return _rotr(x, 2) ^ _rotr(x, 13) ^ _rotr(x, 22)


def _big_sigma1(x):
    return _rotr(x, 6) ^ _rotr(x, 11) ^ _rotr(x, 25)


def _small_sigma0(x):
    return _rotr(x, 7) ^ _rotr(x, 18) ^ _shr(x, 3)


def _small_sigma1(x):
    return _rotr(x, 17) ^ _rotr(x, 19) ^ _shr(x, 10)


# ---------------------------------------------------------------------------
# padding and compression
# ---------------------------------------------------------------------------

def _pad(message_len):
    """The SHA-256 padding for a message of ``message_len`` bytes: 0x80, zeros, then 64-bit bit length."""
    bit_len = message_len * 8
    pad = b"\x80"
    # pad with zeros until length is 56 mod 64, then 8-byte big-endian bit length
    pad += b"\x00" * ((56 - (message_len + 1) % 64) % 64)
    pad += bit_len.to_bytes(8, "big")
    return pad


def _compress(state, block):
    """Fold one 64-byte block into the eight-word state. Returns the new state."""
    w = [0] * 64
    for i in range(16):
        w[i] = int.from_bytes(block[4 * i:4 * i + 4], "big")
    for i in range(16, 64):
        w[i] = (_small_sigma1(w[i - 2]) + w[i - 7] + _small_sigma0(w[i - 15]) + w[i - 16]) & _MASK

    a, b, c, d, e, f, g, h = state
    for i in range(64):
        t1 = (h + _big_sigma1(e) + _ch(e, f, g) + _K[i] + w[i]) & _MASK
        t2 = (_big_sigma0(a) + _maj(a, b, c)) & _MASK
        h = g
        g = f
        f = e
        e = (d + t1) & _MASK
        d = c
        c = b
        b = a
        a = (t1 + t2) & _MASK

    return [(state[i] + v) & _MASK for i, v in enumerate((a, b, c, d, e, f, g, h))]


# ---------------------------------------------------------------------------
# streaming hasher
# ---------------------------------------------------------------------------

class SHA256:
    """Incremental SHA-256 hasher mirroring hashlib's update/digest interface."""

    def __init__(self, data=b""):
        self._state = list(_H0)
        self._buffer = b""
        self._length = 0            # total bytes seen
        if data:
            self.update(data)

    def update(self, data):
        if isinstance(data, str):
            data = data.encode()
        self._length += len(data)
        self._buffer += data
        # process every full 64-byte block, keep the remainder buffered
        while len(self._buffer) >= 64:
            self._state = _compress(self._state, self._buffer[:64])
            self._buffer = self._buffer[64:]
        return self

    def digest(self):
        # finalize on a copy so the hasher can keep being updated
        state = list(self._state)
        tail = self._buffer + _pad(self._length)
        for i in range(0, len(tail), 64):
            state = _compress(state, tail[i:i + 64])
        return b"".join(word.to_bytes(4, "big") for word in state)

    def hexdigest(self):
        return self.digest().hex()


def sha256(data):
    """One-shot SHA-256 digest (bytes) of ``data`` (bytes or str)."""
    return SHA256(data).digest()


def sha256_hex(data):
    """One-shot SHA-256 hex digest of ``data``."""
    return SHA256(data).hexdigest()


# ---------------------------------------------------------------------------
# HMAC-SHA256
# ---------------------------------------------------------------------------

def hmac_sha256(key, message):
    """HMAC-SHA256 keyed message authentication code. Returns the digest bytes."""
    if isinstance(key, str):
        key = key.encode()
    if isinstance(message, str):
        message = message.encode()
    block_size = 64
    if len(key) > block_size:
        key = sha256(key)
    key = key + b"\x00" * (block_size - len(key))
    o_pad = bytes(b ^ 0x5c for b in key)
    i_pad = bytes(b ^ 0x36 for b in key)
    inner = sha256(i_pad + message)
    return sha256(o_pad + inner)


def hmac_sha256_hex(key, message):
    return hmac_sha256(key, message).hex()
