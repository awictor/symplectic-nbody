"""Tests for rabin_karp: rolling-hash search + multi-pattern + LCS vs brute references."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rabin_karp import (search, contains, multi_search, longest_common_substring,
                        brute_search, brute_lcs)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


def random_str(rng, n, alphabet):
    return "".join(alphabet[rng.rand() % len(alphabet)] for _ in range(n))


# --- known cases ------------------------------------------------------------
check("overlapping matches of 'abab'", search("abababab", "abab") == [0, 2, 4])
check("single occurrence", search("hello world", "world") == [6])
check("no occurrence", search("hello", "xyz") == [])
check("pattern longer than text", search("hi", "hello") == [])
check("empty pattern matches every position", search("abc", "") == [0, 1, 2, 3])
check("contains true/false", contains("mississippi", "issi") and not contains("abc", "d"))

# --- search matches brute on random inputs ---------------------------------
rng = LCG(2026)
search_ok = True
for _ in range(500):
    text = random_str(rng, rng.randint(0, 40), "ab")
    m = rng.randint(1, 6)
    pat = random_str(rng, m, "ab")
    if search(text, pat) != brute_search(text, pat):
        search_ok = False
        print(f"  mismatch: text={text!r} pat={pat!r}")
        break
check("Rabin-Karp search matches brute force (500 random pairs)", search_ok)

# --- larger alphabet -------------------------------------------------------
rng = LCG(4242)
alpha_ok = True
for _ in range(300):
    text = random_str(rng, rng.randint(0, 50), "abcdefgh")
    pat = random_str(rng, rng.randint(1, 8), "abcdefgh")
    if search(text, pat) != brute_search(text, pat):
        alpha_ok = False
        break
check("search matches brute over a larger alphabet (300 pairs)", alpha_ok)

# --- multi-pattern search matches per-pattern brute ------------------------
rng = LCG(777)
multi_ok = True
for _ in range(200):
    text = random_str(rng, rng.randint(1, 40), "abc")
    patterns = list({random_str(rng, rng.randint(1, 5), "abc") for _ in range(rng.randint(1, 5))})
    res = multi_search(text, patterns)
    for p in patterns:
        if res[p] != brute_search(text, p):
            multi_ok = False
            break
    if not multi_ok:
        break
check("multi-pattern search matches per-pattern brute force (200 texts)", multi_ok)

# --- longest common substring vs DP ----------------------------------------
rng = LCG(555)
lcs_ok = True
for _ in range(400):
    a = random_str(rng, rng.randint(0, 25), "abc")
    b = random_str(rng, rng.randint(0, 25), "abc")
    got = longest_common_substring(a, b)
    want = brute_lcs(a, b)
    # lengths must match, and the result must be a genuine common substring
    if len(got) != len(want):
        lcs_ok = False
        print(f"  lcs length mismatch: a={a!r} b={b!r} got={got!r} want={want!r}")
        break
    if got and (got not in a or got not in b):
        lcs_ok = False
        break
check("longest common substring length matches the DP reference (400 pairs)", lcs_ok)

# --- known LCS values -------------------------------------------------------
check("LCS of 'banana' and 'ananas' is 'anana'",
      longest_common_substring("banana", "ananas") == "anana")
check("LCS with no overlap is empty", longest_common_substring("abc", "xyz") == "")
check("LCS of a string with itself is the string", longest_common_substring("hello", "hello") == "hello")

# --- occurrences are all valid ---------------------------------------------
rng = LCG(31337)
valid_ok = True
for _ in range(300):
    text = random_str(rng, rng.randint(1, 40), "ab")
    pat = random_str(rng, rng.randint(1, 5), "ab")
    for pos in search(text, pat):
        if text[pos:pos + len(pat)] != pat:
            valid_ok = False
            break
    if not valid_ok:
        break
check("every reported occurrence is a genuine match", valid_ok)

# --- large text performance sanity -----------------------------------------
rng = LCG(99)
big_text = random_str(rng, 20000, "ab")
pat = "abbaab"
check("20k-char text search matches brute", search(big_text, pat) == brute_search(big_text, pat))

# --- repeated single-char text (many overlaps) -----------------------------
check("all-a text finds 'aa' everywhere", search("aaaaa", "aa") == [0, 1, 2, 3])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all rabin_karp tests passed")
