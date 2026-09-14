"""Tests for Ukkonen's suffix tree: containment, occurrence counts, distinct-substring count vs SA+LCP."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from suffix_tree import SuffixTree  # noqa: E402
import suffix_array as SA  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _all_substrings(s):
    return set(s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1))


def _distinct_via_sa(s):
    if not s:
        return 0
    sa = SA.build_suffix_array(s)
    lcp = SA.build_lcp(s, sa)
    n = len(s)
    return n * (n + 1) // 2 - sum(lcp)


def main():
    # ---- 1. banana: containment, counts, distinct, LRS ----------------------------------
    s = "banana"
    t = SuffixTree(s)
    subs = _all_substrings(s)
    check("all real substrings found", all(t.contains(x) for x in subs))
    check("non-substrings rejected",
          not any(t.contains(x) for x in ["xyz", "anab", "bananas", "q"]))
    check("empty string is a substring", t.contains(""))
    for p, exp in [("a", 3), ("na", 2), ("ana", 2), ("ban", 1), ("banana", 1)]:
        check(f"count_occurrences('{p}') == {exp}", t.count_occurrences(p) == exp,
              f"{t.count_occurrences(p)}")
    check("distinct substrings of banana == 15", t.count_distinct_substrings() == 15,
          f"{t.count_distinct_substrings()}")
    check("longest repeated substring of banana == 'ana'",
          t.longest_repeated_substring() == "ana", f"{t.longest_repeated_substring()!r}")

    # ---- 2. distinct-substring count matches the SA+LCP formula -------------------------
    for word in ("mississippi", "abracadabra", "aaaaaa", "abcabcabc", "the quick brown fox"):
        tt = SuffixTree(word)
        dt = tt.count_distinct_substrings()
        ds = _distinct_via_sa(word)
        check(f"distinct('{word[:12]}') == SA+LCP", dt == ds, f"tree {dt} sa {ds}")

    # ---- 3. occurrence counts match brute force -----------------------------------------
    s = "mississippi"
    t = SuffixTree(s)
    ok = True
    for p in _all_substrings(s):
        brute = sum(1 for i in range(len(s)) if s[i:i + len(p)] == p)
        if t.count_occurrences(p) != brute:
            ok = False
            break
    check("occurrence counts match brute force (mississippi)", ok)

    # ---- 4. longest repeated substring matches the max-LCP suffix-array answer ----------
    def lrs_via_sa(s):
        sa = SA.build_suffix_array(s)
        lcp = SA.build_lcp(s, sa)
        best_len = 0
        best_i = 0
        for i in range(len(lcp)):
            if lcp[i] > best_len:
                best_len = lcp[i]
                best_i = sa[i]
        return s[best_i:best_i + best_len]

    for word in ("mississippi", "abcabcabc", "banana", "aabaabaa"):
        t = SuffixTree(word)
        tl = len(t.longest_repeated_substring())
        sl = len(lrs_via_sa(word))
        check(f"LRS length of '{word}' matches SA ({sl})", tl == sl, f"tree {tl} sa {sl}")

    # ---- 5. randomized cross-check on small alphabet ------------------------------------
    rnd = _lcg(99)
    all_ok = True
    for _ in range(40):
        L = int(rnd() * 40) + 5
        s = "".join("abc"[int(rnd() * 3)] for _ in range(L))
        t = SuffixTree(s)
        if t.count_distinct_substrings() != _distinct_via_sa(s):
            all_ok = False
            break
        # spot-check some substrings
        for _ in range(8):
            a = int(rnd() * L)
            b = a + int(rnd() * 6) + 1
            sub = s[a:b]
            if not t.contains(sub):
                all_ok = False
                break
        # a definitely-absent pattern
        if t.contains("d"):
            all_ok = False
            break
    check("40 random strings: distinct==SA+LCP and substrings found", all_ok)

    # ---- 6. single character and empty-ish strings --------------------------------------
    t1 = SuffixTree("a")
    check("single char: 'a' found, 'b' not", t1.contains("a") and not t1.contains("b"))
    check("single char distinct == 1", t1.count_distinct_substrings() == 1)

    # ---- 7. sentinel collision raises ---------------------------------------------------
    try:
        SuffixTree("ab\x00c")
        check("sentinel collision raises", False)
    except ValueError:
        check("sentinel collision raises", True)

    # ---- 8. all-distinct-characters string: n(n+1)/2 substrings -------------------------
    s = "abcdefg"
    t = SuffixTree(s)
    n = len(s)
    check("all-distinct string has n(n+1)/2 substrings",
          t.count_distinct_substrings() == n * (n + 1) // 2, f"{t.count_distinct_substrings()}")
    check("all-distinct string has no repeated substring",
          t.longest_repeated_substring() == "")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
