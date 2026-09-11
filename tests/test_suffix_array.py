"""Tests for suffix_array: sorted-order correctness, LCP, search, longest repeated/common substring."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from suffix_array import (build_suffix_array, build_lcp, search, contains,
                          longest_repeated_substring, longest_common_substring)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def brute_search(s, p):
    if p == "":
        return list(range(len(s) + 1))
    return [i for i in range(len(s) - len(p) + 1) if s[i:i + len(p)] == p]


def sorted_suffix_order(s):
    return sorted(range(len(s)), key=lambda i: s[i:])


# --- suffix array is the true sorted order ---------------------------------
s = "banana"
sa = build_suffix_array(s)
check("banana suffix array", sa == [5, 3, 1, 0, 4, 2])
check("suffix array is the sorted order", sa == sorted_suffix_order(s))
check("suffixes come out sorted", [s[i:] for i in sa] == sorted(s[i:] for i in range(len(s))))

# --- LCP array -------------------------------------------------------------
lcp = build_lcp(s, sa)
check("banana LCP array", lcp == [0, 1, 3, 0, 0, 2])
# LCP definition: lcp[i] is the common prefix length of sa[i-1] and sa[i]
def common_prefix_len(a, b):
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


check("LCP matches direct computation",
      all(lcp[i] == common_prefix_len(s[sa[i - 1]:], s[sa[i]:]) for i in range(1, len(s))))

# --- search ----------------------------------------------------------------
check("search finds all occurrences of 'ana'", search(s, sa, "ana") == [1, 3])
check("search finds 'na'", search(s, sa, "na") == [2, 4])
check("search single char", search(s, sa, "a") == [1, 3, 5])
check("search whole string", search(s, sa, "banana") == [0])
check("search missing pattern", search(s, sa, "xyz") == [])
check("search empty pattern matches everywhere", search(s, sa, "") == list(range(len(s) + 1)))
check("contains true/false", contains(s, sa, "ban") and not contains(s, sa, "xy"))

# --- search agrees with brute force across random strings ------------------
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


all_ok = True
for _ in range(150):
    t = "".join("abc"[int(rng() * 3)] for _ in range(int(rng() * 25) + 1))
    tsa = build_suffix_array(t)
    # sorted-order check
    if tsa != sorted_suffix_order(t):
        all_ok = False
        break
    # LCP check
    tlcp = build_lcp(t, tsa)
    if any(tlcp[i] != common_prefix_len(t[tsa[i - 1]:], t[tsa[i]:]) for i in range(1, len(t))):
        all_ok = False
        break
    # search a random substring
    if len(t) >= 2:
        st = int(rng() * len(t))
        ln = int(rng() * 3) + 1
        p = t[st:st + ln]
        if search(t, tsa, p) != brute_search(t, p):
            all_ok = False
            break
check("suffix array, LCP, and search agree with brute force over 150 strings", all_ok)

# --- longest repeated substring --------------------------------------------
check("LRS of banana is 'ana'", longest_repeated_substring("banana") == "ana")
check("LRS of mississippi is 'issi'", longest_repeated_substring("mississippi") == "issi")
check("LRS with no repeat is empty", longest_repeated_substring("abcdef") == "")
check("LRS of all-same", longest_repeated_substring("aaaa") == "aaa")
check("LRS of a single char is empty", longest_repeated_substring("a") == "")

# a repeated substring really occurs at least twice
for word in ["banana", "mississippi", "abcabcabc", "aabaabaab"]:
    lrs = longest_repeated_substring(word)
    if lrs:
        check(f"LRS of {word} occurs >= twice",
              len(brute_search(word, lrs)) >= 2)

# LRS matches a brute-force longest-repeat over random strings
def brute_lrs(s):
    best = ""
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            sub = s[i:j]
            # count occurrences including overlaps (str.count would miss them)
            if len(sub) > len(best) and len(brute_search(s, sub)) >= 2:
                best = sub
    return best


for _ in range(40):
    t = "".join("ab"[int(rng() * 2)] for _ in range(int(rng() * 14) + 1))
    if len(longest_repeated_substring(t)) != len(brute_lrs(t)):
        check("LRS length matches brute force", False)
        break
else:
    check("LRS length matches brute force over 40 strings", True)

# --- longest common substring ----------------------------------------------
check("LCS of two strings", longest_common_substring("abcdef", "zcdefg") == "cdef")
check("LCS with full overlap", longest_common_substring("hello", "hello") == "hello")
check("LCS with no overlap is empty", longest_common_substring("abc", "xyz") == "")
check("LCS embedded", longest_common_substring("xxGATTACAyy", "zzGATTACAww") == "GATTACA")

# --- empty string ----------------------------------------------------------
check("empty suffix array", build_suffix_array("") == [])
check("empty LCP", build_lcp("", []) == [])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all suffix_array tests passed")
