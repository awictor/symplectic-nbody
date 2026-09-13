"""Tests for Huffman coding: optimal vs brute Kraft enumeration, prefix-free, entropy bracket."""

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huffman import (  # noqa: E402
    build_code,
    canonical_code,
    code_lengths,
    expected_length,
    entropy,
    encode,
    decode,
    is_prefix_free,
    compression_ratio,
    brute_optimal_expected_length,
)


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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def main():
    # ---- 1. Huffman is optimal: matches brute Kraft enumeration -------------------------
    rng = _lcg(2024)
    mism = 0
    for _ in range(150):
        n = 2 + rng() % 6  # 2..7 symbols
        freqs = {i: 1 + rng() % 20 for i in range(n)}
        huff = expected_length(freqs)
        opt = brute_optimal_expected_length(freqs)
        if abs(huff - opt) > 1e-9:
            mism += 1
    check("Huffman expected length == brute optimal (150 alphabets)", mism == 0, f"{mism}")

    # ---- 2. code is prefix-free -------------------------------------------------------
    rng = _lcg(77)
    ok = True
    for _ in range(200):
        n = 1 + rng() % 10
        freqs = {i: 1 + rng() % 30 for i in range(n)}
        if not is_prefix_free(build_code(freqs)):
            ok = False
        if not is_prefix_free(canonical_code(freqs)):
            ok = False
    check("Huffman and canonical codes are prefix-free", ok)

    # ---- 3. round-trip encode/decode ----------------------------------------------------
    rng = _lcg(7)
    ok_h = ok_c = True
    for _ in range(200):
        n = 2 + rng() % 8
        freqs = {chr(ord("a") + i): 1 + rng() % 20 for i in range(n)}
        code = build_code(freqs)
        canon = canonical_code(freqs)
        # random message drawn from the alphabet
        msg = [chr(ord("a") + (rng() % n)) for _ in range(rng() % 40 + 1)]
        if decode(encode(msg, code), code) != msg:
            ok_h = False
        if decode(encode(msg, canon), canon) != msg:
            ok_c = False
    check("Huffman code round-trips messages", ok_h)
    check("canonical code round-trips messages", ok_c)

    # ---- 4. entropy bracket H <= L < H + 1 ----------------------------------------------
    rng = _lcg(555)
    ok = True
    for _ in range(200):
        n = 2 + rng() % 10
        freqs = {i: 1 + rng() % 50 for i in range(n)}
        H = entropy(freqs)
        L = expected_length(freqs)
        if not (H - 1e-9 <= L < H + 1 + 1e-9):
            ok = False
    check("entropy bracket H <= L < H+1 holds", ok)

    # ---- 5. canonical and tree codes have identical lengths -----------------------------
    rng = _lcg(321)
    ok = True
    for _ in range(200):
        n = 1 + rng() % 12
        freqs = {i: 1 + rng() % 40 for i in range(n)}
        if code_lengths(freqs) != {s: len(cw) for s, cw in canonical_code(freqs).items()}:
            ok = False
    check("canonical lengths == tree lengths", ok)

    # ---- 6. dyadic distribution hits entropy exactly ------------------------------------
    # frequencies 1/2, 1/4, 1/8, 1/8 -> entropy 1.75, Huffman length 1.75
    freqs = {"a": 4, "b": 2, "c": 1, "d": 1}
    check("dyadic dist: L == H (1.75)",
          abs(expected_length(freqs) - entropy(freqs)) < 1e-9 and abs(entropy(freqs) - 1.75) < 1e-9,
          f"L={expected_length(freqs)} H={entropy(freqs)}")

    # ---- 7. hand example ----------------------------------------------------------------
    # classic: A=45 B=13 C=12 D=16 E=9 F=5 -> optimal expected length 2.24
    freqs = {"A": 45, "B": 13, "C": 12, "D": 16, "E": 9, "F": 5}
    L = expected_length(freqs)
    check("classic 6-symbol Huffman L ~ 2.24", abs(L - 2.24) < 1e-9, f"{L}")
    check("A (most frequent) has shortest code", len(build_code(freqs)["A"]) == 1)

    # ---- 8. edge cases ------------------------------------------------------------------
    check("empty freqs -> empty code", build_code({}) == {})
    single = build_code({"x": 5})
    check("single symbol code = '0'", single == {"x": "0"})
    check("single symbol round-trips", decode(encode(["x", "x", "x"], single), single) == ["x", "x", "x"])
    # a 3-symbol code has a length-2 codeword; a lone prefix bit of it is undecodable
    c3 = build_code({"a": 4, "b": 1, "c": 1})  # a='1', b='00', c='01'
    check("malformed bits raises", _raises(lambda: decode("0", c3)))  # dangling prefix
    check("compression ratio > 1 for skewed", compression_ratio({"a": 100, "b": 1, "c": 1, "d": 1}) > 1)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


def _raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    main()
