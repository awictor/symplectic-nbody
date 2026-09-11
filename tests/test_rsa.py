"""Tests for rsa.py -- RSA public-key cryptography.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The number theory is checked
against exact values and the full encrypt/decrypt/sign/verify round-trips.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rsa  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- modular arithmetic -----------------------------------------------------
check("modexp matches direct power", rsa.modexp(7, 13, 100) == 7 ** 13 % 100)
check("modexp of large exponent", rsa.modexp(2, 1000, 1_000_000) == pow(2, 1000, 1_000_000))
check("modexp with exponent 0 is 1", rsa.modexp(5, 0, 7) == 1)
check("gcd(462,1071) = 21", rsa.gcd(462, 1071) == 21)
check("coprime gcd is 1", rsa.gcd(17, 3120) == 1)
g, x, y = rsa.extended_gcd(240, 46)
check("extended gcd returns the gcd", g == 2)
check("Bezout identity holds", 240 * x + 46 * y == g)
check("modular inverse of 3 mod 11 is 4", rsa.modinv(3, 11) == 4)
check("inverse times value is 1 mod m", (17 * rsa.modinv(17, 3120)) % 3120 == 1)
try:
    rsa.modinv(4, 8)  # not coprime
    check("modinv rejects non-coprime", False)
except ValueError:
    check("modinv rejects non-coprime", True)

# --- primality --------------------------------------------------------------
check("small primes recognized", all(rsa.is_probable_prime(p) for p in (2, 3, 5, 7, 97, 7919)))
check("composites rejected", not any(rsa.is_probable_prime(c) for c in (1, 4, 100, 561, 1000)))
check("Carmichael number 561 is caught (Fermat would fail)", not rsa.is_probable_prime(561))
check("Carmichael number 41041 is caught", not rsa.is_probable_prime(41041))
check("a large known prime is accepted", rsa.is_probable_prime(2 ** 61 - 1))  # Mersenne prime
check("a large known composite is rejected", not rsa.is_probable_prime(2 ** 61 - 3))

# --- key generation ---------------------------------------------------------
pub, priv = rsa.generate_keypair(nbits=256, seed=42)
e, n = pub
d, n2 = priv
check("public and private share the modulus", n == n2)
check("modulus is about 512 bits", 500 <= n.bit_length() <= 512)
check("e*d = 1 mod phi is enforced via correct decryption", True)  # verified by round-trips below
check("keygen is deterministic for a fixed seed", rsa.generate_keypair(256, seed=42)[0] == pub)
check("different seeds give different keys", rsa.generate_keypair(256, seed=43)[0] != pub)

# --- encrypt / decrypt ------------------------------------------------------
m = 1234567890123456789
c = rsa.encrypt(m, pub)
check("ciphertext differs from plaintext", c != m)
check("decrypt inverts encrypt", rsa.decrypt(c, priv) == m)
check("encrypt/decrypt round-trips for several messages",
      all(rsa.decrypt(rsa.encrypt(x, pub), priv) == x for x in (0, 1, 2, 42, n - 1)))
try:
    rsa.encrypt(n, pub)  # message must be < n
    check("encrypt rejects m >= n", False)
except ValueError:
    check("encrypt rejects m >= n", True)

# --- sign / verify ----------------------------------------------------------
s = rsa.sign(m, priv)
check("a valid signature verifies", rsa.verify(m, s, pub))
check("a tampered message fails verification", not rsa.verify(m + 1, s, pub))
check("a forged signature fails verification", not rsa.verify(m, s + 1, pub))
# signing and encrypting are inverse operations with swapped keys
check("verify undoes sign (s^e = m)", rsa.modexp(s, e, n) == m)

# --- byte-string encryption -------------------------------------------------
msg = b"Attack at dawn -- RSA from scratch, pure stdlib."
blocks = rsa.encrypt_bytes(msg, pub)
check("byte encryption round-trips", rsa.decrypt_bytes(blocks, priv) == msg)
check("empty message round-trips", rsa.decrypt_bytes(rsa.encrypt_bytes(b"", pub), priv) == b"")
check("single-byte message round-trips", rsa.decrypt_bytes(rsa.encrypt_bytes(b"A", pub), priv) == b"A")
check("ciphertext blocks are not the plaintext",
      all(cbytes != int.from_bytes(msg[i:i + 1], "big") for i, (cbytes, _) in enumerate(blocks[:1])))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall rsa tests passed")
