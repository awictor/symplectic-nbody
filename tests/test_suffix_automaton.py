"""Tests for suffix_automaton: distinct substrings, membership, occurrences, LRS, LCS vs brute force."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from suffix_automaton import (build, count_distinct_substrings, longest_repeated_substring,
                              longest_common_substring, brute_distinct_substrings,
                              brute_longest_common_substring)

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


def brute_occurrences(s, pat):
    if not pat:
        return 0
    count = 0
    for i in range(len(s) - len(pat) + 1):
        if s[i:i + len(pat)] == pat:
            count += 1
    return count


# --- known cases ------------------------------------------------------------
check("distinct substrings of 'abcbc' is 12", count_distinct_substrings("abcbc") == 12)
check("distinct substrings of 'banana' is 15", count_distinct_substrings("banana") == 15)
check("distinct substrings of 'aaaa' is 4", count_distinct_substrings("aaaa") == 4)
check("empty string has 0 distinct substrings", count_distinct_substrings("") == 0)

sa = build("abcbc")
check("'bcb' is a substring", sa.contains("bcb"))
check("'abc' is a substring", sa.contains("abc"))
check("'xyz' is not a substring", not sa.contains("xyz"))
check("empty pattern is trivially contained", sa.contains(""))

check("longest repeated substring of 'banana' is 'ana'", longest_repeated_substring("banana") == "ana")
check("no repeat -> empty LRS", longest_repeated_substring("abcd") == "")

check("LCS of 'abcbc' and 'xbcby' is 'bcb'", longest_common_substring("abcbc", "xbcby") == "bcb")

# --- distinct-substring count vs brute -------------------------------------
rng = LCG(2026)
distinct_ok = True
for _ in range(400):
    n = rng.randint(0, 12)
    s = random_string(rng, n, "abc") if n else ""
    if count_distinct_substrings(s) != len(brute_distinct_substrings(s)):
        distinct_ok = False
        print(f"  distinct mismatch for {s!r}")
        break
check("distinct-substring count matches brute force (400 random strings)", distinct_ok)

# --- membership vs brute ----------------------------------------------------
rng = LCG(4242)
member_ok = True
for _ in range(200):
    s = random_string(rng, rng.randint(1, 12), "ab")
    sa = build(s)
    # test all substrings (should be present) and some random patterns (may not be)
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if not sa.contains(s[i:j]):
                member_ok = False
                break
    for _ in range(5):
        pat = random_string(rng, rng.randint(1, 5), "abc")
        if sa.contains(pat) != (pat in s):
            member_ok = False
            break
    if not member_ok:
        break
check("substring membership matches direct scanning (all substrings + random patterns)", member_ok)

# --- occurrence counts vs brute --------------------------------------------
rng = LCG(777)
occ_ok = True
for _ in range(200):
    s = random_string(rng, rng.randint(1, 14), "ab")
    sa = build(s)
    for _ in range(8):
        pat = random_string(rng, rng.randint(1, 5), "ab")
        if sa.occurrences(pat) != brute_occurrences(s, pat):
            occ_ok = False
            print(f"  occ mismatch: s={s!r} pat={pat!r} sam={sa.occurrences(pat)} "
                  f"brute={brute_occurrences(s, pat)}")
            break
    if not occ_ok:
        break
check("occurrence counts match brute scanning (200 strings x 8 patterns)", occ_ok)

# every substring of s occurs at least once
rng = LCG(555)
occ_pos_ok = True
for _ in range(100):
    s = random_string(rng, rng.randint(1, 10), "ab")
    sa = build(s)
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if sa.occurrences(s[i:j]) < 1:
                occ_pos_ok = False
                break
check("every substring occurs at least once", occ_pos_ok)

# --- longest common substring vs DP ----------------------------------------
rng = LCG(31337)
lcs_ok = True
for _ in range(300):
    s = random_string(rng, rng.randint(0, 12), "abc")
    t = random_string(rng, rng.randint(0, 12), "abc")
    got = longest_common_substring(s, t)
    want = brute_longest_common_substring(s, t)
    # lengths must match, and the returned string must be a common substring
    if len(got) != len(want):
        lcs_ok = False
        print(f"  LCS length mismatch: s={s!r} t={t!r} got={got!r} want={want!r}")
        break
    if got and (got not in s or got not in t):
        lcs_ok = False
        break
check("longest common substring length matches the DP reference (300 pairs)", lcs_ok)

# --- state-count bound: at most 2n-1 states --------------------------------
rng = LCG(99)
bound_ok = True
for _ in range(100):
    n = rng.randint(2, 30)
    s = random_string(rng, n, "ab")
    sa = build(s)
    if len(sa.states) > 2 * n:            # 2n-1 states + the initial state accounting
        bound_ok = False
        break
check("state count stays within the linear 2n bound", bound_ok)

# --- longest repeated substring is genuinely repeated ----------------------
rng = LCG(1234)
lrs_ok = True
for _ in range(200):
    s = random_string(rng, rng.randint(1, 14), "ab")
    lrs = longest_repeated_substring(s)
    if lrs:
        if brute_occurrences(s, lrs) < 2:
            lrs_ok = False
            break
        # no longer repeated substring exists
        longer_repeat = False
        L = len(lrs) + 1
        for i in range(len(s) - L + 1):
            if brute_occurrences(s, s[i:i + L]) >= 2:
                longer_repeat = True
                break
        if longer_repeat:
            lrs_ok = False
            break
check("longest repeated substring is repeated and maximal (200 strings)", lrs_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all suffix_automaton tests passed")
