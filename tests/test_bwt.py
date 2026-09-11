"""Tests for bwt: transform roundtrip, runniness gain, MTF, RLE, full bzip2-style pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bwt import (bwt_transform, bwt_inverse, move_to_front_encode, move_to_front_decode,
                 rle_encode, rle_decode, compress, decompress, bwt_runniness_gain,
                 _mean_run_length)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- BWT round-trips any string --------------------------------------------
cases = ["", "a", "banana", "abracadabra", "mississippi", "the quick brown fox jumps",
         "aaaaaa", "abcabcabc", "hello world hello world", "!@#$%^&*()", "112233445566"]
check("BWT round-trips every test string",
      all(bwt_inverse(bwt_transform(s)) == s for s in cases))

# --- BWT is a permutation (same multiset of characters, plus a sentinel) ---
t = bwt_transform("banana")
check("BWT output is one longer (sentinel added)", len(t) == len("banana") + 1)
check("BWT preserves the character multiset (minus sentinel)",
      sorted(t.replace("\x00", "")) == sorted("banana"))

# --- the classic banana transform -----------------------------------------
# BWT(banana + sentinel) is a known permutation; verify it inverts to banana
check("banana inverts correctly", bwt_inverse(bwt_transform("banana")) == "banana")

# --- BWT increases run length on structured text ---------------------------
structured = "the theme of these theses is theory " * 4
before, after = bwt_runniness_gain(structured)
check("BWT increases the mean run length", after > before)
check("mean run length of a uniform string is its length", _mean_run_length("aaaa") == 4.0)
check("mean run length of an alternating string is 1", _mean_run_length("abababab") == 1.0)

# --- move-to-front round-trips ---------------------------------------------
codes, alpha = move_to_front_encode("banana")
check("MTF round-trips", move_to_front_decode(codes, alpha) == "banana")
# a run of one symbol MTF-encodes to a run of zeros after the first
codes_run, alpha_run = move_to_front_encode("aaaaa")
check("MTF turns a repeat into zeros", codes_run == [0, 0, 0, 0, 0])
check("MTF codes are alphabet indices", all(0 <= c < len(alpha) for c in codes))

# --- run-length round-trips ------------------------------------------------
seq = [1, 1, 1, 2, 2, 3, 3, 3, 3]
pairs = rle_encode(seq)
check("RLE encodes runs", pairs == [(1, 3), (2, 2), (3, 4)])
check("RLE round-trips", rle_decode(pairs) == seq)
check("RLE of empty is empty", rle_encode([]) == [])
check("RLE of a single value", rle_encode([5]) == [(5, 1)])

# --- the full pipeline is lossless -----------------------------------------
for s in ["abracadabra", "mississippi" * 5, "hello world " * 3,
          "the theme of theses" * 4, "aaaaaaaaaa", "abcdefg"]:
    check(f"pipeline round-trips {s[:12]!r}", decompress(*compress(s)) == s)

# --- the pipeline shrinks repetitive input ---------------------------------
repetitive = "ababababab" * 12
pairs_r, alpha_r = compress(repetitive)
check("pipeline yields far fewer RLE pairs than input length", len(pairs_r) < len(repetitive) / 4)

# a run-heavy input compresses even more
runs = "aaaaabbbbbcccccddddd" * 6
pairs_runs, _ = compress(runs)
check("run-heavy input compresses strongly", len(pairs_runs) < len(runs) / 3)

# --- random-ish text still round-trips (stress) ----------------------------
state = 12345


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


alphabet = "abcdefg "
for _ in range(50):
    length = 1 + int(rng() * 60)
    s = "".join(alphabet[int(rng() * len(alphabet))] for _ in range(length))
    if decompress(*compress(s)) != s:
        check("random text round-trips", False)
        break
else:
    check("50 random strings round-trip through the pipeline", True)

# --- MTF then inverse over a BWT output ------------------------------------
bw = bwt_transform("mississippi")
c, a = move_to_front_encode(bw)
check("MTF over a BWT output round-trips", move_to_front_decode(c, a) == bw)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bwt tests passed")
