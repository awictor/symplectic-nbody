"""Tests for sha256: bit-for-bit agreement with hashlib, NIST vectors, streaming, HMAC vs hmac."""

import hashlib
import hmac as _hmac
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sha256 import (SHA256, sha256, sha256_hex, hmac_sha256, hmac_sha256_hex,  # noqa: E402
                    _rotr, _ch, _maj)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def byte(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 16) & 0xFF


def main():
    # ---- 1. known NIST / classic vectors ----------------------------------------------
    check("empty string digest",
          sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    check("'abc' digest",
          sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    check("two-block message digest",
          sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
          == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")

    # ---- 2. agreement with hashlib on every length 0..200 -----------------------------
    rng = LCG(2024)
    mism = 0
    for n in range(201):
        data = bytes(rng.byte() for _ in range(n))
        if sha256(data) != hashlib.sha256(data).digest():
            mism += 1
    check("matches hashlib for every length 0..200 (padding boundaries)", mism == 0,
          f"{mism} mismatches")

    # ---- 3. hundreds of random strings ------------------------------------------------
    mism_r = 0
    for _ in range(500):
        n = rng.byte() * 3
        data = bytes(rng.byte() for _ in range(n))
        if sha256_hex(data) != hashlib.sha256(data).hexdigest():
            mism_r += 1
    check("matches hashlib on 500 random inputs", mism_r == 0, f"{mism_r} mismatches")

    # ---- 4. streaming equals one-shot regardless of chunking --------------------------
    data = bytes(rng.byte() for _ in range(1000))
    one_shot = sha256(data)
    stream_bad = 0
    for chunk in (1, 7, 63, 64, 65, 128, 333):
        h = SHA256()
        for i in range(0, len(data), chunk):
            h.update(data[i:i + chunk])
        if h.digest() != one_shot:
            stream_bad += 1
    check("streaming update matches one-shot for all chunk sizes", stream_bad == 0,
          f"{stream_bad} chunk sizes disagreed")

    # digest() can be called mid-stream without breaking further updates
    h = SHA256()
    h.update(b"hello ")
    _ = h.digest()          # peek
    h.update(b"world")
    check("digest() is non-destructive", h.digest() == sha256(b"hello world"))

    # ---- 5. avalanche: one bit flip changes about half the output bits ----------------
    a = sha256(b"avalanche test message")
    b = sha256(b"avalanche test messagf")   # last byte differs by one bit region
    diff_bits = sum(bin(a[i] ^ b[i]).count("1") for i in range(32))
    check("one-character change flips ~half the 256 bits", 96 < diff_bits < 160,
          f"{diff_bits} bits differ")

    # ---- 6. string input accepted -----------------------------------------------------
    check("str input hashes same as its utf-8 bytes", sha256("abc") == sha256(b"abc"))

    # ---- 7. HMAC-SHA256 vs the standard library ---------------------------------------
    hmac_bad = 0
    for _ in range(200):
        klen = rng.byte() % 100
        mlen = rng.byte() % 150
        key = bytes(rng.byte() for _ in range(klen))
        msg = bytes(rng.byte() for _ in range(mlen))
        mine = hmac_sha256(key, msg)
        ref = _hmac.new(key, msg, hashlib.sha256).digest()
        if mine != ref:
            hmac_bad += 1
    check("HMAC-SHA256 matches hmac stdlib on 200 random pairs", hmac_bad == 0,
          f"{hmac_bad} mismatches")

    # long key (> block size) path
    longkey = bytes(rng.byte() for _ in range(100))
    check("HMAC with long key matches stdlib",
          hmac_sha256(longkey, b"msg") == _hmac.new(longkey, b"msg", hashlib.sha256).digest())

    # RFC 4231 test case 1
    key = bytes.fromhex("0b" * 20)
    check("HMAC RFC 4231 vector 1",
          hmac_sha256_hex(key, b"Hi There")
          == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7")

    # ---- 8. primitives ----------------------------------------------------------------
    check("rotr wraps correctly", _rotr(1, 1) == 0x80000000)
    check("ch selects by x", _ch(0xFFFFFFFF, 0xAAAAAAAA, 0x55555555) == 0xAAAAAAAA)
    # majority bit-by-bit: bit set iff at least two inputs have it
    check("maj is bitwise majority",
          _maj(0b110, 0b101, 0b011) == 0b111 and _maj(0b100, 0b010, 0b001) == 0b000)
    check("maj all-equal", _maj(0x1234, 0x1234, 0x1234) == 0x1234)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
