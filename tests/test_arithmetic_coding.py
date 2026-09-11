"""Tests for arithmetic_coding: roundtrip, approaches entropy, beats Huffman on skew, edge cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arithmetic_coding import (encode, decode, entropy, encoded_bits_per_symbol, build_model)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def roundtrip(msg):
    bits, model, total, length = encode(msg)
    return "".join(decode(bits, model, total, length))


# --- frequency model -------------------------------------------------------
model, total = build_model("aab")
check("model total is the length", total == 3)
check("model partitions [0,total)", model["a"] == (0, 2) and model["b"] == (2, 3))

# --- entropy ---------------------------------------------------------------
check("entropy of a fair coin sequence is 1", abs(entropy("abab") - 1.0) < 1e-9)
check("entropy of a constant string is 0", entropy("aaaa") == 0.0)
check("entropy of 4 equal symbols is 2", abs(entropy("abcd") - 2.0) < 1e-9)

# --- round-trip a variety of messages --------------------------------------
messages = ["abracadabra", "mississippi", "hello world", "aaaaaa", "a",
            "the quick brown fox jumps over the lazy dog", "112233", "!@#$%^"]
check("round-trips every test message", all(roundtrip(m) == m for m in messages))

# --- approaches the Shannon entropy ----------------------------------------
skewed = "a" * 90 + "b" * 7 + "c" * 3
H = entropy(skewed)
ac = encoded_bits_per_symbol(skewed)
check("arithmetic code length approaches entropy", abs(ac - H) < 0.1)
# total overhead is a small constant (a couple of bits for the whole message)
bits, _, _, length = encode(skewed)
check("total overhead over entropy is tiny", len(bits) - H * length < 3.0)

# --- beats Huffman on a skewed distribution --------------------------------
# Huffman must use >= 1 bit per symbol (three symbols); AC uses far less here
check("arithmetic coding beats the 1-bit-per-symbol Huffman floor", ac < 1.0)

# a more extreme skew -> even fewer bits per symbol
very_skewed = "a" * 995 + "b" * 5
check("extreme skew compresses below 0.1 bits/symbol", encoded_bits_per_symbol(very_skewed) < 0.1)

# --- uniform alphabet costs about log2(k) bits -----------------------------
uni = "abcd" * 30
check("uniform 4-symbol alphabet ~ 2 bits/symbol", abs(encoded_bits_per_symbol(uni) - 2.0) < 0.1)
uni8 = "abcdefgh" * 20
check("uniform 8-symbol alphabet ~ 3 bits/symbol", abs(encoded_bits_per_symbol(uni8) - 3.0) < 0.1)

# --- single-symbol alphabet (entropy 0) ------------------------------------
single = "aaaaaaaaaa"
check("single-symbol message round-trips", roundtrip(single) == single)
check("single-symbol message is tiny", len(encode(single)[0]) < 40)

# --- empty message ---------------------------------------------------------
check("empty message round-trips", roundtrip("") == "")
check("empty encode gives empty bits", encode("")[0] == "")

# --- decode consumes only the model + bits (determinism) -------------------
b1 = encode("mississippi")
b2 = encode("mississippi")
check("encoding is deterministic", b1 == b2)

# --- a longer structured message, exact round-trip -------------------------
long_msg = ("the theme of these theses is theory " * 5)
check("long structured message round-trips", roundtrip(long_msg) == long_msg)
check("structured message compresses below its entropy-free size",
      encoded_bits_per_symbol(long_msg) < math.log2(len(set(long_msg))) + 0.2)

# --- random stress: every message round-trips ------------------------------
state = 7


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


all_ok = True
for _ in range(60):
    n = 1 + int(rng() * 80)
    m = "".join("abcde"[int(rng() * 5)] for _ in range(n))
    if roundtrip(m) != m:
        all_ok = False
        break
check("60 random messages round-trip exactly", all_ok)

# --- code length is never absurdly larger than entropy predicts -----------
for m in messages:
    if len(m) >= 4:
        H_m = entropy(m)
        actual = encoded_bits_per_symbol(m)
        # within entropy + a couple of bits amortized over the message
        check(f"{m[:8]!r} code length is near-optimal",
              actual <= H_m + 3.0 / len(m) + 0.05 or actual <= H_m + 0.5)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all arithmetic_coding tests passed")
