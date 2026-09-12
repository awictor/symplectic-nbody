"""Tests for eertree: distinct palindrome count, palindrome list, occurrence counts vs brute force."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eertree import (build, count_distinct_palindromes, brute_distinct_palindromes,
                     brute_occurrence_counts, is_palindrome)

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


def random_string(rng, n, alphabet):
    return "".join(alphabet[rng.rand() % len(alphabet)] for _ in range(n))


# --- known counts -----------------------------------------------------------
check("distinct palindromes of 'aba' is 3 (a, b, aba)", count_distinct_palindromes("aba") == 3)
check("distinct palindromes of 'abacaba' is 7", count_distinct_palindromes("abacaba") == 7)
check("distinct palindromes of 'aaaa' is 4 (a, aa, aaa, aaaa)",
      count_distinct_palindromes("aaaa") == 4)
check("empty string has 0 palindromes", count_distinct_palindromes("") == 0)
check("single char has 1 palindrome", count_distinct_palindromes("x") == 1)
check("all-distinct string 'abcde' has 5 palindromes (the singletons)",
      count_distinct_palindromes("abcde") == 5)

# --- distinct count vs brute -----------------------------------------------
rng = LCG(2026)
distinct_ok = True
for _ in range(500):
    s = random_string(rng, rng.randint(0, 16), "ab") if rng.rand() % 2 else \
        random_string(rng, rng.randint(0, 16), "abc")
    if count_distinct_palindromes(s) != len(brute_distinct_palindromes(s)):
        distinct_ok = False
        print(f"  distinct mismatch for {s!r}")
        break
check("distinct palindrome count matches brute force (500 random strings)", distinct_ok)

# --- palindrome SET matches brute ------------------------------------------
rng = LCG(4242)
set_ok = True
for _ in range(300):
    s = random_string(rng, rng.randint(1, 14), "abc")
    et = build(s)
    got = set("".join(p) for p in et.palindromes())
    want = brute_distinct_palindromes(s)
    if got != want:
        set_ok = False
        print(f"  palindrome set mismatch for {s!r}")
        break
    # every reconstructed palindrome really is a palindrome and a substring
    if not all(is_palindrome(p) and "".join(p) in s for p in et.palindromes()):
        set_ok = False
        break
check("the reconstructed palindrome set matches brute enumeration (300 strings)", set_ok)

# --- occurrence counts vs brute --------------------------------------------
rng = LCG(777)
occ_ok = True
for _ in range(300):
    s = random_string(rng, rng.randint(1, 14), "ab")
    et = build(s)
    got = {"".join(k): v for k, v in et.occurrence_counts().items()}
    want = brute_occurrence_counts(s)
    if got != want:
        occ_ok = False
        print(f"  occurrence mismatch: s={s!r}\n    got={got}\n    want={want}")
        break
check("palindrome occurrence counts match direct scanning (300 strings)", occ_ok)

# --- the classical bound: at most n distinct palindromic substrings --------
rng = LCG(555)
bound_ok = True
for _ in range(300):
    n = rng.randint(0, 20)
    s = random_string(rng, n, "abc")
    if count_distinct_palindromes(s) > n:
        bound_ok = False
        break
check("a string has at most n distinct palindromic substrings (classical bound)", bound_ok)

# a string that achieves the maximum n (e.g. all same char)
check("'aaaaa' achieves the bound with exactly 5 palindromes",
      count_distinct_palindromes("aaaaa") == 5)

# --- online build equals batch build ---------------------------------------
rng = LCG(99)
online_ok = True
for _ in range(100):
    s = random_string(rng, rng.randint(1, 15), "ab")
    batch = build(s).count_distinct()
    et = build("")
    for ch in s:
        et.add(ch)
    if et.count_distinct() != batch:
        online_ok = False
        break
check("incremental add() matches batch construction", online_ok)

# --- occurrence totals: sum of (occurrences) equals total palindromic substrings
rng = LCG(31337)
total_ok = True
for _ in range(200):
    s = random_string(rng, rng.randint(1, 13), "ab")
    et = build(s)
    total = sum(et.occurrence_counts().values())
    # brute: total number of palindromic substrings (with multiplicity)
    brute_total = sum(1 for i in range(len(s)) for j in range(i + 1, len(s) + 1)
                      if is_palindrome(s[i:j]))
    if total != brute_total:
        total_ok = False
        break
check("summed occurrence counts equal the total number of palindromic substrings", total_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all eertree tests passed")
