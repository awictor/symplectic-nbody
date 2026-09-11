"""Tests for levenshtein.py -- edit distance and alignment.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Distances are checked against
known values, the metric axioms, and a brute-force check that the alignment reproduces the
target with exactly `distance` edits.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import levenshtein as L  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- known distances --------------------------------------------------------
check("kitten -> sitting is 3", L.distance("kitten", "sitting") == 3)
check("flaw -> lawn is 2", L.distance("flaw", "lawn") == 2)
check("Saturday -> Sunday is 3", L.distance("Saturday", "Sunday") == 3)
check("identical strings have distance 0", L.distance("abc", "abc") == 0)
check("distance to empty is the length", L.distance("hello", "") == 5 and L.distance("", "hi") == 2)
check("both empty is 0", L.distance("", "") == 0)
check("single substitution", L.distance("cat", "bat") == 1)
check("single insertion", L.distance("cat", "cart") == 1)
check("single deletion", L.distance("cart", "cat") == 1)

# --- fast (two-row) version agrees -----------------------------------------
pairs = [("kitten", "sitting"), ("flaw", "lawn"), ("", "x"), ("abcdef", "azced"),
         ("aaa", "aaa"), ("longer string here", "a longer strung her")]
check("distance_fast agrees with the full table",
      all(L.distance(a, b) == L.distance_fast(a, b) for a, b in pairs))

# --- metric axioms ----------------------------------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 16


g = lcg(1)


def rand_str(maxlen=8, alpha="abcd"):
    n = next(g) % (maxlen + 1)
    return "".join(alpha[next(g) % len(alpha)] for _ in range(n))


sym_ok = tri_ok = zero_ok = True
for _ in range(500):
    a, b, c = rand_str(), rand_str(), rand_str()
    if L.distance(a, b) != L.distance(b, a):
        sym_ok = False
    if L.distance(a, c) > L.distance(a, b) + L.distance(b, c):
        tri_ok = False
    if (L.distance(a, b) == 0) != (a == b):
        zero_ok = False
check("distance is symmetric", sym_ok)
check("distance obeys the triangle inequality", tri_ok)
check("distance is 0 iff the strings are equal", zero_ok)

# --- alignment reproduces the target with exactly `distance` edits ---------
align_ok = True
for a, b in pairs:
    ops = L.alignment(a, b)
    if L.apply_ops(a, ops) != b:
        align_ok = False
        break
    edits = sum(1 for o in ops if o[0] != "match")
    if edits != L.distance(a, b):
        align_ok = False
        break
check("alignment reproduces the target and uses exactly `distance` edits", align_ok)
# random alignment check
for _ in range(300):
    a, b = rand_str(), rand_str()
    ops = L.alignment(a, b)
    if L.apply_ops(a, ops) != b or sum(1 for o in ops if o[0] != "match") != L.distance(a, b):
        align_ok = False
        break
check("alignment is correct over 300 random pairs", align_ok)
# op types are well-formed
ops = L.alignment("kitten", "sitting")
check("alignment ops are valid kinds",
      all(o[0] in ("match", "substitute", "delete", "insert") for o in ops))

# --- similarity -------------------------------------------------------------
check("identical strings have similarity 1", L.similarity("abc", "abc") == 1.0)
check("two empty strings have similarity 1", L.similarity("", "") == 1.0)
check("completely different strings have similarity 0", L.similarity("aaa", "bbb") == 0.0)
check("similarity is in [0, 1]", 0.0 <= L.similarity("kitten", "sitting") <= 1.0)
check("more similar strings score higher",
      L.similarity("cat", "cot") > L.similarity("cat", "dog"))

# --- Damerau transposition --------------------------------------------------
check("Levenshtein counts a transposition as 2", L.distance("teh", "the") == 2)
check("Damerau counts a transposition as 1", L.damerau_distance("teh", "the") == 1)
check("Damerau agrees with Levenshtein when no transposition helps",
      L.damerau_distance("kitten", "sitting") == L.distance("kitten", "sitting"))
check("Damerau <= Levenshtein always",
      all(L.damerau_distance(a, b) <= L.distance(a, b) for a, b in pairs))
check("Damerau of 'ca' -> 'ac' is 1 (one swap)", L.damerau_distance("ca", "ac") == 1)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall levenshtein tests passed")
