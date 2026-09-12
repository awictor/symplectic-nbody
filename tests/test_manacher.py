"""Tests for manacher: longest palindrome and count vs brute force, radii, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manacher import (longest_palindrome, count_palindromic_substrings, all_palindrome_radii,
                      is_palindrome)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 12321
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_longest(s):
    best = ""
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            sub = s[i:j]
            if sub == sub[::-1] and len(sub) > len(best):
                best = sub
    return best


def brute_count(s):
    return sum(1 for i in range(len(s)) for j in range(i + 1, len(s) + 1)
               if s[i:j] == s[i:j][::-1])


# --- known cases -----------------------------------------------------------
check("babad longest is bab or aba", longest_palindrome("babad") in ("bab", "aba"))
check("cbbd longest is bb", longest_palindrome("cbbd") == "bb")
check("racecar longest is racecar", longest_palindrome("racecar") == "racecar")
check("aaaa longest is aaaa", longest_palindrome("aaaa") == "aaaa")
check("abc longest is a single char", len(longest_palindrome("abc")) == 1)

# --- longest matches brute force over random strings -----------------------
ok_len = ok_count = True
for _ in range(500):
    n = int(rng() * 20)
    s = "".join(chr(ord("a") + int(rng() * 3)) for _ in range(n))     # small alphabet -> palindromes
    m_long = longest_palindrome(s)
    b_long = brute_longest(s)
    # same length (there may be ties); and the returned substring is actually a palindrome
    if len(m_long) != len(b_long) or m_long != m_long[::-1]:
        ok_len = False
        break
    if m_long and m_long not in s:
        ok_len = False
        break
    if count_palindromic_substrings(s) != brute_count(s):
        ok_count = False
        break
check("longest palindrome matches brute force over 500 random strings", ok_len)
check("palindrome count matches brute force over 500 random strings", ok_count)

# --- count on known strings ------------------------------------------------
check("count('aaa') = 6", count_palindromic_substrings("aaa") == 6)   # a,a,a,aa,aa,aaa
check("count('abc') = 3", count_palindromic_substrings("abc") == 3)   # each single char
check("count('aba') = 4", count_palindromic_substrings("aba") == 4)   # a,b,a,aba

# --- returned palindrome is always a real palindrome and a substring -------
ok = True
for _ in range(200):
    s = "".join(chr(ord("a") + int(rng() * 4)) for _ in range(int(rng() * 30)))
    lp = longest_palindrome(s)
    if lp and (lp != lp[::-1] or lp not in s):
        ok = False
        break
check("returned longest palindrome is genuine and a substring", ok)

# --- radii: odd and even palindrome radii are consistent -------------------
odd, even = all_palindrome_radii("abacaba")
# center at 'c' (index 3) should have the largest odd radius (abacaba is a full palindrome)
check("center of a symmetric string has the maximal odd radius", odd[3] == max(odd))
# radii describe genuine palindromes
def verify_radii(s):
    odd, even = all_palindrome_radii(s)
    for i, r in enumerate(odd):
        # odd palindrome of radius r centered at i spans [i-(r-1), i+(r-1)]
        lo, hi = i - (r - 1), i + (r - 1)
        if s[lo:hi + 1] != s[lo:hi + 1][::-1]:
            return False
    for i, r in enumerate(even):
        if r > 0:
            lo, hi = i - r + 1, i + r
            if s[lo:hi + 1] != s[lo:hi + 1][::-1]:
                return False
    return True

check("odd/even radii describe genuine palindromes", verify_radii("abacabadabacaba"))

# --- edge cases ------------------------------------------------------------
check("empty string: empty longest, zero count",
      longest_palindrome("") == "" and count_palindromic_substrings("") == 0)
check("single char: itself, count 1",
      longest_palindrome("x") == "x" and count_palindromic_substrings("x") == 1)
check("all distinct: count equals length (only single chars)",
      count_palindromic_substrings("abcdef") == 6)
check("all same: longest is the whole string",
      longest_palindrome("zzzzz") == "zzzzz")

# --- is_palindrome ---------------------------------------------------------
check("is_palindrome true", is_palindrome("racecar") and is_palindrome("abba"))
check("is_palindrome false", not is_palindrome("abc"))
check("is_palindrome empty and single", is_palindrome("") and is_palindrome("a"))

# --- a long palindrome inside noise ----------------------------------------
s = "xyz" + "abcdefggfedcba" + "qrs"
check("finds an embedded palindrome", longest_palindrome(s) == "abcdefggfedcba")

# --- performance sanity: a long all-same string (worst case for naive) -----
big = "a" * 2000
lp = longest_palindrome(big)
check("linear-time handles a 2000-char all-same string", lp == big)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all manacher tests passed")
