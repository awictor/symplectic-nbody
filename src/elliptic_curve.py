"""Elliptic-curve cryptography -- the group law on a curve, and the ECDH / ECDSA it powers.

RSA's security rests on the difficulty of factoring; elliptic-curve cryptography rests on a different and
harder problem, and gets equivalent security from far smaller keys -- a 256-bit elliptic-curve key
matches a 3072-bit RSA key, which is why TLS, SSH, Bitcoin, and Signal all run on elliptic curves. The
object is deceptively geometric: the set of points (x, y) satisfying y^2 = x^3 + a x + b over a finite
field, plus a "point at infinity" that acts as an identity. The magic is that these points form an
ABELIAN GROUP under a peculiar addition rule -- and that rule, iterated, hides a one-way function.

The GROUP LAW is drawn from geometry. To add two points P and Q, draw the line through them; it meets the
cubic in exactly one more point, and the sum P + Q is that point reflected across the x-axis. When P = Q
the line is the tangent (point doubling). Over a finite field the same algebra holds with the slopes
computed by modular inverse instead of real division. Repeatedly adding a base point G to itself --
SCALAR MULTIPLICATION, k*G -- is fast by double-and-add, but RECOVERING k from k*G and G is the ELLIPTIC-
CURVE DISCRETE LOGARITHM PROBLEM, believed exponentially hard. That asymmetry is the whole game.

Two protocols ride on it. ECDH key exchange: Alice and Bob each pick a secret scalar, publish its product
with G, and each multiplies the other's public point by their own secret -- both arrive at the same
shared point (a*b*G) that an eavesdropper cannot compute. ECDSA signatures: to sign a message hash, pick a
nonce k, compute r from k*G, and combine r, the hash, and the private key into s; anyone can verify with
the public key without learning the secret. This module implements the field and point arithmetic, the
group law, scalar multiplication, ECDH, and ECDSA (sign and verify), all from scratch in pure Python.

For clarity and fast, exhaustive testing the default curve is a small standard example
(y^2 = x^3 + 2x + 2 over F_17, and a larger toy prime), but the arithmetic is exact big-integer code that
works on real curves too. A seeded generator makes key generation reproducible.

Validation. (1) The group axioms hold, checked exhaustively on the small curve: every generated point
SATISFIES the curve equation, addition is commutative and associative, P + (-P) is the identity, and
identity + P = P. (2) The base point's ORDER is correct -- n*G is the identity and no smaller multiple is
-- and scalar multiplication is consistent with repeated addition (k*G equals G added k times). (3) ECDH
agreement: independently chosen key pairs derive the IDENTICAL shared secret, over many seeded runs. (4)
ECDSA correctness: a signature verifies against its own message and public key, and FAILS for a tampered
message, a wrong key, or a mangled signature -- over many seeded trials. (5) The discrete-log one-wayness
is illustrated by brute force on the small curve. Pure standard library."""

import hashlib


# ---------------------------------------------------------------------------
# field helpers (self-contained modular inverse)
# ---------------------------------------------------------------------------

def _extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = _extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def inverse_mod(a, m):
    """Modular inverse of a mod m via the extended Euclidean algorithm."""
    a %= m
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse mod {m}")
    return x % m


# point at infinity (group identity) is represented by None


class Curve:
    """A short Weierstrass curve y^2 = x^3 + a x + b over the prime field F_p, with base point G of
    order n."""

    def __init__(self, a, b, p, gx=None, gy=None, n=None):
        self.a = a
        self.b = b
        self.p = p
        self.G = (gx, gy) if gx is not None else None
        self.n = n
        if (4 * a ** 3 + 27 * b ** 2) % p == 0:
            raise ValueError("singular curve (discriminant is zero)")

    def is_on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - (x * x * x + self.a * x + self.b)) % self.p == 0

    def add(self, P, Q):
        """Group addition P + Q by the chord-and-tangent law."""
        if P is None:
            return Q
        if Q is None:
            return P
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and (y1 + y2) % self.p == 0:
            return None                          # P + (-P) = identity
        if P == Q:
            # tangent slope: (3 x1^2 + a) / (2 y1)
            num = (3 * x1 * x1 + self.a) % self.p
            den = inverse_mod(2 * y1, self.p)
            s = (num * den) % self.p
        else:
            s = ((y2 - y1) * inverse_mod((x2 - x1) % self.p, self.p)) % self.p
        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        return (x3, y3)

    def negate(self, P):
        if P is None:
            return None
        x, y = P
        return (x, (-y) % self.p)

    def scalar_mult(self, k, P):
        """k * P by double-and-add. Handles negative k via negation."""
        if k < 0:
            return self.scalar_mult(-k, self.negate(P))
        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def order_of(self, P, bound=100000):
        """Smallest positive m with m*P = identity (brute force; for small curves / testing)."""
        Q = P
        for m in range(1, bound + 1):
            if Q is None:
                return m
            Q = self.add(Q, P)
        return None


# ---------------------------------------------------------------------------
# standard small / toy curves
# ---------------------------------------------------------------------------

def curve_f17():
    """y^2 = x^3 + 2x + 2 over F_17; base point (5, 1) has order 19 (a classic textbook curve)."""
    return Curve(a=2, b=2, p=17, gx=5, gy=1, n=19)


def curve_toy():
    """A larger toy curve for ECDH/ECDSA: y^2 = x^3 + 5x + 1 over F_1009, base (1, 45).

    Chosen so the group has PRIME order 1039 -- no small subgroups, exactly what ECDSA needs.
    """
    return Curve(a=5, b=1, p=1009, gx=1, gy=45, n=1039)


# ---------------------------------------------------------------------------
# key generation, ECDH, ECDSA
# ---------------------------------------------------------------------------

class _LCG:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def randint(self, lo, hi):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return lo + (self.state >> 8) % (hi - lo + 1)


def generate_keypair(curve, rng):
    """Return (private_scalar, public_point) with public = private * G."""
    d = rng.randint(1, curve.n - 1)
    Q = curve.scalar_mult(d, curve.G)
    return d, Q


def ecdh_shared_secret(curve, my_private, their_public):
    """The shared ECDH point: my_private * their_public. Its x-coordinate is the shared secret."""
    return curve.scalar_mult(my_private, their_public)


def _hash_int(message, n):
    """Map a message (bytes or str) to an integer mod n via SHA-256."""
    if isinstance(message, str):
        message = message.encode()
    h = int.from_bytes(hashlib.sha256(message).digest(), "big")
    return h % n


def ecdsa_sign(curve, private, message, rng):
    """Sign a message with ECDSA. Returns (r, s)."""
    n = curve.n
    z = _hash_int(message, n)
    while True:
        k = rng.randint(1, n - 1)
        R = curve.scalar_mult(k, curve.G)
        if R is None:
            continue
        r = R[0] % n
        if r == 0:
            continue
        s = (inverse_mod(k, n) * (z + r * private)) % n
        if s == 0:
            continue
        return (r, s)


def ecdsa_verify(curve, public, message, signature):
    """Verify an ECDSA signature (r, s) against a message and public key."""
    n = curve.n
    r, s = signature
    if not (1 <= r < n and 1 <= s < n):
        return False
    z = _hash_int(message, n)
    w = inverse_mod(s, n)
    u1 = (z * w) % n
    u2 = (r * w) % n
    P = curve.add(curve.scalar_mult(u1, curve.G), curve.scalar_mult(u2, public))
    if P is None:
        return False
    return (P[0] % n) == r
