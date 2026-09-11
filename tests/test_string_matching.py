"""Tests for string_matching: prefix function, KMP, Z-algorithm, Manacher, brute-force agreement."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from string_matching import (prefix_function, kmp_search, z_array, z_search,
                             brute_force_search, longest_palindrome, count_occurrences)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- prefix function against its definition --------------------------------
check("prefix function of ababaca", prefix_function("ababaca") == [0, 0, 1, 2, 3, 0, 1])
check("prefix function of aaaa", prefix_function("aaaa") == [0, 1, 2, 3])
check("prefix function of abcabc", prefix_function("abcabc") == [0, 0, 0, 1, 2, 3])
check("prefix function of a distinct string is all zeros", prefix_function("abcd") == [0, 0, 0, 0])

# prefix function definition holds: s[:pi[i]] == s[i-pi[i]+1:i+1]
for s in ["ababaca", "aabaabaab", "mississippi"]:
    pi = prefix_function(s)
    check(f"prefix definition holds for {s}",
          all(s[:pi[i]] == s[i - pi[i] + 1:i + 1] for i in range(len(s))))

# --- KMP search ------------------------------------------------------------
check("KMP finds all matches", kmp_search("abcabcabc", "abc") == [0, 3, 6])
check("KMP finds overlapping matches", kmp_search("aaaa", "aa") == [0, 1, 2])
check("KMP no match", kmp_search("abcdef", "xyz") == [])
check("KMP whole-string match", kmp_search("hello", "hello") == [0])
check("KMP single char", kmp_search("banana", "a") == [1, 3, 5])
check("KMP empty pattern matches everywhere", kmp_search("ab", "") == [0, 1, 2])
check("KMP pattern longer than text", kmp_search("ab", "abc") == [])

# --- Z-array ---------------------------------------------------------------
check("Z-array of aabaab", z_array("aabaab") == [0, 1, 0, 3, 1, 0])
check("Z-array of aaaa", z_array("aaaa") == [0, 3, 2, 1])
check("Z-array of abcabc", z_array("abcabc") == [0, 0, 0, 3, 0, 0])

# --- Z-search agrees with KMP ----------------------------------------------
check("Z-search matches KMP on overlaps", z_search("aaaa", "aa") == [0, 1, 2])
check("Z-search whole match", z_search("hello", "hello") == [0])

# --- KMP == Z == brute force across many strings ---------------------------
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


all_agree = True
for _ in range(400):
    tlen = int(rng() * 30) + 1
    plen = int(rng() * 5) + 1
    t = "".join("abc"[int(rng() * 3)] for _ in range(tlen))
    p = "".join("abc"[int(rng() * 3)] for _ in range(plen))
    k = kmp_search(t, p)
    z = z_search(t, p)
    b = brute_force_search(t, p)
    if not (k == z == b):
        all_agree = False
        break
check("KMP, Z, and brute force agree over 400 random strings", all_agree)

# --- count occurrences -----------------------------------------------------
check("count overlapping occurrences", count_occurrences("aaaaa", "aa") == 4)
check("count non-overlapping", count_occurrences("abcabc", "abc") == 2)
check("count zero", count_occurrences("abc", "d") == 0)

# --- Manacher longest palindrome -------------------------------------------
check("babad -> bab or aba", longest_palindrome("babad") in ("bab", "aba"))
check("cbbd -> bb", longest_palindrome("cbbd") == "bb")
check("racecar is its own palindrome", longest_palindrome("racecar") == "racecar")
check("embedded palindrome", longest_palindrome("forgeeksskeegfor") == "geeksskeeg")
check("all-same is fully palindromic", longest_palindrome("aaaa") == "aaaa")
check("no palindrome > 1 returns a single char", len(longest_palindrome("abcd")) == 1)
check("empty string", longest_palindrome("") == "")
check("single char", longest_palindrome("z") == "z")
check("even palindrome", longest_palindrome("abba") == "abba")

# --- the returned palindrome really is a palindrome and a substring --------
for s in ["babad", "forgeeksskeegfor", "abacabad", "noonracecar"]:
    pal = longest_palindrome(s)
    check(f"result for {s} is a palindrome", pal == pal[::-1])
    check(f"result for {s} is a substring", pal in s)

# --- Manacher matches a brute-force longest palindrome ---------------------
def brute_palindrome(s):
    best = ""
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            sub = s[i:j]
            if sub == sub[::-1] and len(sub) > len(best):
                best = sub
    return best


for _ in range(100):
    s = "".join("ab"[int(rng() * 2)] for _ in range(int(rng() * 15) + 1))
    if len(longest_palindrome(s)) != len(brute_palindrome(s)):
        check("Manacher matches brute-force length", False)
        break
else:
    check("Manacher length matches brute force over 100 strings", True)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all string_matching tests passed")
