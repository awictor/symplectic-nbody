"""Tests for rans: exact round-trip on every input, compression near Shannon entropy, model sums to M."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rans import (encode, decode, build_model, entropy, bits_per_symbol, _M)  # noqa: E402


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

    def nxt(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        return lo + (self.nxt() >> 8) % (hi - lo + 1)


def roundtrip(data):
    enc, model = encode(data)
    return decode(enc, model, len(data)) == data


def main():
    # ---- 1. hand cases ----------------------------------------------------------------
    for d in (b"", b"a", b"aa", b"aaaa", b"abracadabra", b"the quick brown fox",
              bytes([0] * 200), bytes(range(256))):
        check(f"round trip {d[:16]!r} (len {len(d)})", roundtrip(d))

    # ---- 2. round trip on hundreds of random byte strings, varied alphabets -----------
    rng = LCG(2024)
    bad = 0
    for _ in range(500):
        n = rng.randint(0, 400)
        alpha = rng.randint(1, 16)
        data = bytes(rng.randint(0, alpha - 1) for _ in range(n))
        if not roundtrip(data):
            bad += 1
    check("round trip on 500 random strings", bad == 0, f"{bad} failures")

    # ---- 3. skewed distributions round-trip -------------------------------------------
    skew_bad = 0
    for trial in range(50):
        rng2 = LCG(trial * 13 + 1)
        # 85% symbol 0, rest spread over 1..4
        data = bytes(0 if rng2.randint(0, 99) < 85 else rng2.randint(1, 4) for _ in range(500))
        if not roundtrip(data):
            skew_bad += 1
    check("round trip on 50 skewed streams", skew_bad == 0, f"{skew_bad} failures")

    # ---- 4. compression approaches Shannon entropy ------------------------------------
    rng = LCG(7)
    # a strongly skewed source
    data = bytes(0 if rng.randint(0, 99) < 80 else rng.randint(1, 6) for _ in range(5000))
    H = entropy(data)
    bps = bits_per_symbol(data)
    check("encoded bits/symbol below 8 (compresses)", bps < 8, f"{bps:.3f} bps")
    check("encoded bits/symbol near entropy (within 5%)", bps <= H * 1.05 + 0.05,
          f"bps {bps:.3f} vs H {H:.3f}")
    check("encoded bits/symbol not below entropy (Shannon limit)", bps >= H - 0.05,
          f"bps {bps:.3f} vs H {H:.3f}")

    # ---- 5. near-uniform source stays near log2(alphabet) -----------------------------
    rng = LCG(99)
    uni = bytes(rng.randint(0, 15) for _ in range(4000))    # 16 symbols ~ 4 bits each
    check("uniform 16-symbol source ~ 4 bits/symbol", abs(bits_per_symbol(uni) - 4.0) < 0.3,
          f"{bits_per_symbol(uni):.3f}")

    # ---- 6. single repeated symbol compresses to almost nothing -----------------------
    mono = bytes([42] * 10000)
    enc, model = encode(mono)
    check("monotone data compresses hugely", len(enc) < 20, f"{len(enc)} bytes for 10000")
    check("monotone round trip", decode(enc, model, len(mono)) == mono)

    # ---- 7. frequency model sums exactly to M -----------------------------------------
    for seed in range(10):
        r = LCG(seed + 1)
        data = bytes(r.randint(0, 20) for _ in range(300))
        freq, cum, symbols = build_model(data)
        if symbols:
            total = sum(freq[s] for s in symbols)
            if total != _M:
                check("model frequencies sum to M", False, f"sum {total} != {_M}")
                break
            # cumulative is consistent
            c = 0
            ok = True
            for s in symbols:
                if cum[s] != c:
                    ok = False
                c += freq[s]
            if not ok:
                check("cumulative frequencies consistent", False)
                break
    else:
        check("model frequencies sum to M and cumulative is consistent", True)

    # ---- 8. entropy of monotone data is 0 ---------------------------------------------
    check("entropy of single-symbol data is 0", entropy(bytes([5] * 100)) == 0.0)
    check("entropy of uniform 2-symbol data is 1", abs(entropy(bytes([0, 1] * 100)) - 1.0) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
