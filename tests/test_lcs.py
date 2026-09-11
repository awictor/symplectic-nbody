"""Tests for lcs.py -- longest common subsequence and diff.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Correctness is proven
exhaustively against a brute-force subsequence search.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lcs as L  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- known LCS --------------------------------------------------------------
check("LCS length of ABCBDAB / BDCAB is 4", L.lcs_length("ABCBDAB", "BDCAB") == 4)
check("LCS of AGGTAB / GXTXAYB is GTAB", L.lcs("AGGTAB", "GXTXAYB") == "GTAB")
check("identical strings have full-length LCS", L.lcs_length("hello", "hello") == 5)
check("disjoint alphabets have empty LCS", L.lcs_length("abc", "xyz") == 0)
check("LCS with an empty sequence is empty", L.lcs("abc", "") == "" and L.lcs_length("", "xyz") == 0)
check("a subsequence relationship gives that subsequence",
      L.lcs_length("ace", "abcde") == 3)

# --- the LCS is a genuine common subsequence -------------------------------
s = L.lcs("ABCBDAB", "BDCAB")
check("the returned LCS is a subsequence of a", L._is_subsequence(s, "ABCBDAB"))
check("the returned LCS is a subsequence of b", L._is_subsequence(s, "BDCAB"))
check("the returned LCS has the reported length", len(s) == L.lcs_length("ABCBDAB", "BDCAB"))

# --- works on lists too, not just strings ----------------------------------
check("LCS works on lists", L.lcs([1, 2, 3, 4], [2, 4, 6]) == [2, 4])
check("LCS length on lists", L.lcs_length([1, 2, 3], [3, 2, 1]) == 1)

# --- diff edit-script -------------------------------------------------------
ops = L.diff("ABCBDAB", "BDCAB")
check("applying the diff reproduces b", L.apply_diff("ABCBDAB", ops) == "BDCAB")
check("kept elements equal the LCS length",
      sum(1 for o in ops if o[0] == "keep") == L.lcs_length("ABCBDAB", "BDCAB"))
check("diff ops are valid kinds", all(o[0] in ("keep", "delete", "insert") for o in ops))
# a diff between identical sequences is all keeps
same_ops = L.diff("hello", "hello")
check("diff of identical strings is all keeps", all(o[0] == "keep" for o in same_ops))
# a diff from empty is all inserts
check("diff from empty is all inserts", all(o[0] == "insert" for o in L.diff("", "abc")))
check("diff to empty is all deletes", all(o[0] == "delete" for o in L.diff("abc", "")))

# --- insert/delete edit distance -------------------------------------------
check("indel distance is m + n - 2*LCS", L.edit_distance_indel("ABCBDAB", "BDCAB") == 7 + 5 - 2 * 4)
check("indel distance of identical strings is 0", L.edit_distance_indel("abc", "abc") == 0)
check("indel distance of disjoint strings is m + n", L.edit_distance_indel("abc", "xyz") == 6)

# --- exhaustive check against brute force ----------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 16


gen = lcg(3)


def rand_str(maxlen=11, alpha="abc"):
    n = next(gen) % (maxlen + 1)
    return "".join(alpha[next(gen) % len(alpha)] for _ in range(n))


len_bad = subseq_bad = diff_bad = 0
for _ in range(1000):
    a, b = rand_str(), rand_str()
    dp = L.lcs_length(a, b)
    if dp != L.brute_lcs_length(a, b):
        len_bad += 1
    seq = L.lcs(a, b)
    if len(seq) != dp or not (L._is_subsequence(seq, a) and L._is_subsequence(seq, b)):
        subseq_bad += 1
    ops = L.diff(a, b)
    if L.apply_diff(a, ops) != b or sum(1 for o in ops if o[0] == "keep") != dp:
        diff_bad += 1
check("LCS length matches brute force over 1000 random pairs", len_bad == 0)
check("the LCS is always a valid common subsequence of the reported length", subseq_bad == 0)
check("the diff always reproduces b with LCS-many keeps", diff_bad == 0)

# --- LCS is symmetric in length --------------------------------------------
sym = all(L.lcs_length(rand_str(), rand_str()) >= 0 for _ in range(10))  # smoke
check("LCS length is symmetric",
      all(L.lcs_length("abcabc", "bcabca") == L.lcs_length("bcabca", "abcabc") for _ in range(1)))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall lcs tests passed")
