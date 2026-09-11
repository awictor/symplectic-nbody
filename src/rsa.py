"""RSA: public-key cryptography from the hardness of factoring.

Rivest, Shamir, and Adleman's 1977 scheme was the first practical way for two strangers to
communicate secretly without ever sharing a key in advance. Its security rests on a beautiful
asymmetry: multiplying two large primes p and q into n = pq is easy, but recovering p and q from
n alone is (as far as anyone knows) astronomically hard.

The construction:

  * pick two primes p, q and set n = pq and the totient phi = (p-1)(q-1);
  * choose a public exponent e coprime to phi (commonly 65537);
  * compute the private exponent d = e^{-1} mod phi (the modular inverse, via extended Euclid).

Encryption raises the message to the public exponent, decryption to the private one:

    c = m^e mod n,     m = c^d mod n,

and the two undo each other because e d = 1 mod phi, so by Euler's theorem m^{ed} = m mod n.
The same pair also SIGNS: encrypting a hash with the private key produces a signature anyone can
verify with the public key. Modular exponentiation is done by fast square-and-multiply, so even
huge exponents cost only ~log2(e) multiplications.

This module implements Miller-Rabin primality testing, the extended Euclidean algorithm and
modular inverse, RSA key generation, textbook encrypt/decrypt and sign/verify, all in pure
stdlib big integers. It is an educational reference -- real deployments add OAEP/PSS padding --
but the number theory is exactly the real thing. The public-key companion to the Hamming and
Shannon notes.
"""

from __future__ import annotations


def extended_gcd(a: int, b: int):
    """Return (g, x, y) with g = gcd(a, b) and a*x + b*y = g (Bezout coefficients)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def gcd(a: int, b: int) -> int:
    """Greatest common divisor via the Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)


def modinv(a: int, m: int) -> int:
    """Modular multiplicative inverse of a modulo m: the x with a*x = 1 (mod m). Raises if a and
    m are not coprime."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError("no modular inverse: a and m are not coprime")
    return x % m


def modexp(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation base^exp mod mod by square-and-multiply."""
    if exp < 0:
        raise ValueError("exponent must be nonnegative")
    result = 1
    base %= mod
    while exp:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result


# --- primality (Miller-Rabin) ----------------------------------------------

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def is_probable_prime(n: int, witnesses=None) -> bool:
    """Miller-Rabin primality test. Deterministic for n < 3.3e24 with the standard first-12
    prime witnesses; probabilistic (but overwhelmingly reliable) beyond that."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # write n-1 = d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (witnesses or _SMALL_PRIMES):
        if a % n == 0:
            continue
        x = modexp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random). For reproducible key gen only;
    NOT a cryptographically secure source -- a real system uses os.urandom."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def bits(self, nbits: int) -> int:
        """A random nbits-bit integer with the top bit set (so it is exactly nbits long)."""
        val = 1
        for _ in range(nbits - 1):
            val = (val << 1) | ((self._next() >> 16) & 1)
        return val | (1 << (nbits - 1))


def generate_prime(nbits: int, rng: _Rng) -> int:
    """Generate a probable prime of about nbits bits using the given (seeded) RNG."""
    while True:
        cand = rng.bits(nbits) | 1  # force odd
        if is_probable_prime(cand):
            return cand


def generate_keypair(nbits: int = 256, e: int = 65537, seed: int = 1):
    """Generate an RSA keypair. Returns ((e, n) public, (d, n) private). nbits is the size of
    each prime, so the modulus is about 2*nbits bits."""
    rng = _Rng(seed)
    p = generate_prime(nbits, rng)
    q = generate_prime(nbits, rng)
    while q == p:
        q = generate_prime(nbits, rng)
    n = p * q
    phi = (p - 1) * (q - 1)
    if gcd(e, phi) != 1:
        raise ValueError("e is not coprime to phi; choose another e")
    d = modinv(e, phi)
    return (e, n), (d, n)


def encrypt(message: int, public_key) -> int:
    """Encrypt an integer message (0 <= m < n) with the public key (e, n): c = m^e mod n."""
    e, n = public_key
    if not 0 <= message < n:
        raise ValueError("message must satisfy 0 <= m < n")
    return modexp(message, e, n)


def decrypt(cipher: int, private_key) -> int:
    """Decrypt a ciphertext with the private key (d, n): m = c^d mod n."""
    d, n = private_key
    return modexp(cipher, d, n)


def sign(message: int, private_key) -> int:
    """Sign an integer message with the private key: s = m^d mod n (textbook, no hashing)."""
    d, n = private_key
    if not 0 <= message < n:
        raise ValueError("message must satisfy 0 <= m < n")
    return modexp(message, d, n)


def verify(message: int, signature: int, public_key) -> bool:
    """Verify a signature against the public key: check s^e mod n == m."""
    e, n = public_key
    return modexp(signature, e, n) == (message % n)


def encrypt_bytes(data: bytes, public_key):
    """Encrypt a byte string by chunking it into integers smaller than n. Returns a list of
    ciphertext integers. Educational: chunk size is one byte less than the modulus size."""
    e, n = public_key
    chunk = (n.bit_length() - 1) // 8  # bytes per block, strictly below n
    if chunk < 1:
        raise ValueError("modulus too small to encrypt bytes")
    out = []
    for i in range(0, len(data), chunk):
        block = data[i:i + chunk]
        m = int.from_bytes(block, "big")
        out.append((encrypt(m, public_key), len(block)))
    return out


def decrypt_bytes(blocks, private_key) -> bytes:
    """Decrypt a list of (ciphertext, length) blocks produced by encrypt_bytes back to bytes."""
    out = bytearray()
    for c, length in blocks:
        m = decrypt(c, private_key)
        out.extend(m.to_bytes(length, "big"))
    return bytes(out)
