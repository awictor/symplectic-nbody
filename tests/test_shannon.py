"""Tests for shannon.py -- Shannon entropy and Huffman coding.

Self-running: prints PASS/FAIL per check, exits 1 if any fail.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import shannon as s  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- entropy ----------------------------------------------------------------
check("fair coin has entropy 1 bit", approx(s.entropy([0.5, 0.5]), 1.0, 1e-12))
check("a certain outcome has entropy 0", approx(s.entropy([1.0, 0.0]), 0.0, 1e-12))
check("uniform 4-symbol source has entropy 2 bits", approx(s.entropy([0.25] * 4), 2.0, 1e-12))
check("uniform n-source entropy is log2(n)", approx(s.entropy([0.125] * 8), 3.0, 1e-12))
check("zero-probability symbols contribute nothing", approx(s.entropy([0.5, 0.5, 0.0]), 1.0, 1e-12))
check("entropy is maximal for the uniform distribution",
      s.entropy([0.25] * 4) > s.entropy([0.7, 0.1, 0.1, 0.1]))
check("max_entropy is log2(n)", approx(s.max_entropy(8), 3.0, 1e-12))
check("entropy never exceeds the maximum", s.entropy([0.7, 0.1, 0.1, 0.1]) <= s.max_entropy(4) + 1e-12)
check("entropy from counts matches probabilities",
      approx(s.entropy_from_counts([1, 1, 1, 1]), 2.0, 1e-12))
try:
    s.entropy([0.5, -0.1])
    check("rejects negative probabilities", False)
except ValueError:
    check("rejects negative probabilities", True)

# --- Huffman code structure ------------------------------------------------
freqs = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
code = s.huffman_code(freqs)
check("every symbol gets a codeword", set(code.keys()) == set(freqs.keys()))
check("Huffman code is prefix-free", s.is_prefix_free(code))
check("Kraft sum is <= 1", s.kraft_sum(code) <= 1.0 + 1e-12)
check("Kraft sum is 1 for a complete code", approx(s.kraft_sum(code), 1.0, 1e-12))
check("more frequent symbol has a no-longer codeword", len(code["a"]) <= len(code["d"]))
# single symbol edge case
solo = s.huffman_code({"x": 5})
check("a single symbol codes to one bit", solo == {"x": "0"})

# --- the Shannon source-coding bound ---------------------------------------
H = s.entropy([0.4, 0.3, 0.2, 0.1])
L = s.average_length(code, freqs)
check("average length is at least the entropy (L >= H)", L >= H - 1e-12)
check("average length is below H + 1", L < H + 1.0)
check("efficiency H/L is in (0, 1]", 0 < s.efficiency(code, freqs) <= 1.0 + 1e-12)
# a dyadic distribution (powers of 1/2) is coded with efficiency exactly 1
dyadic = {"a": 0.5, "b": 0.25, "c": 0.125, "d": 0.125}
dcode = s.huffman_code(dyadic)
check("dyadic source is coded at 100% efficiency", approx(s.efficiency(dcode, dyadic), 1.0, 1e-9))
check("dyadic average length equals the entropy",
      approx(s.average_length(dcode, dyadic), s.entropy([0.5, 0.25, 0.125, 0.125]), 1e-9))
# uniform source: all codewords equal length, L = log2(n)
uni = {c: 1 for c in "abcd"}
ucode = s.huffman_code(uni)
check("uniform source uses fixed-length 2-bit codes",
      all(len(c) == 2 for c in ucode.values()))

# Huffman is optimal: no prefix code beats its average length. Compare against a naive
# fixed-length code on the skewed source.
fixed_len = math.ceil(math.log2(len(freqs)))
check("Huffman beats or ties the fixed-length code on a skewed source", L <= fixed_len)

# --- lossless round-trip ----------------------------------------------------
message = list("abracadabra")
freq_msg = {}
for ch in message:
    freq_msg[ch] = freq_msg.get(ch, 0) + 1
mcode = s.huffman_code(freq_msg)
bits = s.encode(message, mcode)
check("encode/decode round-trips exactly", s.decode(bits, mcode) == message)
check("encoded stream is shorter than 8 bits/char", len(bits) < 8 * len(message))
# the encoded length equals sum of codeword lengths
check("encoded length matches the sum of codeword lengths",
      len(bits) == sum(len(mcode[c]) for c in message))
try:
    s.decode(bits + "1", mcode)  # dangling partial codeword
    check("decode rejects a truncated stream", False)
except ValueError:
    check("decode rejects a truncated stream", True)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall shannon tests passed")
