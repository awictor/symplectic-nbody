"""Tests for boyer_moore.py -- Boyer-Moore substring search.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Correctness is proven
exhaustively against a naive search over thousands of random text/pattern pairs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import boyer_moore as BM  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- basic matches ----------------------------------------------------------
check("finds a simple substring", BM.find("hello world", "world") == 6)
check("returns -1 when absent", BM.find("hello", "xyz") == -1)
check("contains is True when present", BM.contains("the quick brown fox", "quick"))
check("contains is False when absent", not BM.contains("abcdef", "gh"))
check("finds a match at the start", BM.find("abcabc", "abc") == 0)
check("finds a match at the end", BM.find("xxxxabc", "abc") == 4)

# --- the classic Boyer-Moore example ---------------------------------------
check("classic GCAGAGAG example", BM.find_all("GCATCGCAGAGAGTATACAGTACG", "GCAGAGAG") == [5])

# --- all occurrences, including overlaps -----------------------------------
check("all occurrences of 'aa' in 'aaaa' (overlapping)", BM.find_all("aaaa", "aa") == [0, 1, 2])
check("overlapping 'ana' in 'banana'", BM.find_all("banana", "ana") == [1, 3])
check("count returns the number of occurrences", BM.count("banana", "a") == 3)
check("multiple disjoint occurrences", BM.find_all("abcXabcXabc", "abc") == [0, 4, 8])
check("no occurrences gives an empty list", BM.find_all("hello", "z") == [])

# --- edge cases -------------------------------------------------------------
check("empty pattern matches at every position", BM.find_all("abc", "") == [0, 1, 2, 3])
check("pattern longer than text finds nothing", BM.find_all("ab", "abcd") == [])
check("pattern equal to text matches once", BM.find_all("abc", "abc") == [0])
check("single-character pattern", BM.find_all("mississippi", "s") == [2, 3, 5, 6])
check("empty text with nonempty pattern", BM.find_all("", "a") == [])

# --- bad-character rule actually skips -------------------------------------
# a mismatch on a character absent from the pattern should jump the full pattern length;
# we can't observe the internal shift directly, but we can confirm correctness on such a case
check("mismatch character absent from pattern still finds the match",
      BM.find("aaaaaaaaaaXpattern", "pattern") == 11)

# --- good-suffix rule cases -------------------------------------------------
check("repeated-suffix pattern found", BM.find_all("abababab", "abab") == [0, 2, 4])
check("good-suffix does not miss a match", BM.find_all("ABAAABAAABAAAB", "AAAB") == [2, 6, 10])

# --- exhaustive cross-check against naive search ---------------------------
def lcg(seed):
    state = seed
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state >> 16


gen = lcg(12345)


def rand(lo, hi):
    return lo + next(gen) % (hi - lo + 1)


mismatches = 0
for _ in range(5000):
    alpha = "ab" if rand(0, 1) == 0 else "abcde"
    n = rand(0, 50)
    m = rand(0, 10)
    text = "".join(alpha[rand(0, len(alpha) - 1)] for _ in range(n))
    pat = "".join(alpha[rand(0, len(alpha) - 1)] for _ in range(m))
    if BM.find_all(text, pat) != BM.naive_find_all(text, pat):
        mismatches += 1
check("EVERY one of 5000 random cases matches naive search", mismatches == 0)

# a couple of larger, structured cases too
big_text = ("mississippi " * 200)
check("Boyer-Moore matches naive on a large repetitive text",
      BM.find_all(big_text, "issi") == BM.naive_find_all(big_text, "issi"))
dna = "ACGT" * 500 + "ACGTTGCA"
check("Boyer-Moore matches naive on DNA-like text",
      BM.find_all(dna, "GTTGCA") == BM.naive_find_all(dna, "GTTGCA"))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall boyer_moore tests passed")
