"""Tests for combinatorial_rank: permutation/combination rank-unrank bijections, Gray code."""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from combinatorial_rank import (permutation_rank, permutation_unrank, combination_rank,
                                combination_unrank, gray_encode, gray_decode, gray_sequence)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- permutation ranks match itertools lexicographic order -----------------
ok = True
for n in range(1, 8):
    perms = list(itertools.permutations(range(n)))
    for i, p in enumerate(perms):
        if permutation_rank(list(p)) != i:
            ok = False
            break
    if not ok:
        break
check("permutation_rank matches itertools order for n up to 7", ok)

# --- permutation rank/unrank is a bijection --------------------------------
ok = True
for n in range(1, 8):
    fact = math.factorial(n)
    seen = set()
    for r in range(fact):
        p = permutation_unrank(r, n)
        if sorted(p) != list(range(n)):     # must be a valid permutation
            ok = False
            break
        if permutation_rank(p) != r:         # round-trip
            ok = False
            break
        seen.add(tuple(p))
    if len(seen) != fact:                    # all distinct
        ok = False
    if not ok:
        break
check("permutation rank/unrank is a bijection over 0..n!-1", ok)

# --- combination ranks match itertools.combinations order ------------------
ok = True
for n in range(1, 9):
    for k in range(n + 1):
        combos = list(itertools.combinations(range(n), k))
        for i, c in enumerate(combos):
            if combination_rank(list(c), n) != i:
                ok = False
                break
        if not ok:
            break
    if not ok:
        break
check("combination_rank matches itertools.combinations order", ok)

# --- combination rank/unrank is a bijection --------------------------------
ok = True
for n in range(1, 9):
    for k in range(n + 1):
        total = math.comb(n, k)
        seen = set()
        for r in range(total):
            c = combination_unrank(r, n, k)
            if len(c) != k or sorted(c) != c or any(x < 0 or x >= n for x in c):
                ok = False
                break
            if combination_rank(c, n) != r:
                ok = False
                break
            seen.add(tuple(c))
        if len(seen) != total:
            ok = False
        if not ok:
            break
    if not ok:
        break
check("combination rank/unrank is a bijection over 0..C(n,k)-1", ok)

# --- Gray code: consecutive codes differ in exactly one bit ----------------
ok = True
for bits in range(1, 11):
    seq = gray_sequence(bits)
    if len(seq) != (1 << bits):
        ok = False
        break
    if len(set(seq)) != (1 << bits):        # all distinct (a permutation of 0..2^bits-1)
        ok = False
        break
    for i in range(len(seq)):
        nxt = seq[(i + 1) % len(seq)]        # also wraps around (cyclic Gray code)
        if bin(seq[i] ^ nxt).count("1") != 1:
            ok = False
            break
    if not ok:
        break
check("Gray code: consecutive (and wraparound) codes differ in one bit", ok)

# --- Gray encode/decode round-trip -----------------------------------------
check("gray_decode(gray_encode(n)) == n over 0..1000",
      all(gray_decode(gray_encode(n)) == n for n in range(1001)))

# --- known values ----------------------------------------------------------
check("gray_encode(0..7) is the standard sequence",
      [gray_encode(n) for n in range(8)] == [0, 1, 3, 2, 6, 7, 5, 4])
check("first permutation of 4 is identity", permutation_unrank(0, 4) == [0, 1, 2, 3])
check("last permutation of 4 is reversed", permutation_unrank(23, 4) == [3, 2, 1, 0])
check("first 2-combination of 5 is {0,1}", combination_unrank(0, 5, 2) == [0, 1])
check("last 2-combination of 5 is {3,4}", combination_unrank(math.comb(5, 2) - 1, 5, 2) == [3, 4])

# --- ranks are exactly 0..N-1 (contiguous, no gaps) ------------------------
n = 6
ranks = sorted(permutation_rank(list(p)) for p in itertools.permutations(range(n)))
check("permutation ranks are exactly 0..n!-1", ranks == list(range(math.factorial(n))))

n, k = 8, 3
ranks = sorted(combination_rank(list(c), n) for c in itertools.combinations(range(n), k))
check("combination ranks are exactly 0..C(n,k)-1", ranks == list(range(math.comb(n, k))))

# --- unranking a random large index (no full enumeration needed) -----------
# the 1,000,000-th permutation of 10 elements
p = permutation_unrank(1000000, 10)
check("large-index permutation is valid and round-trips",
      sorted(p) == list(range(10)) and permutation_rank(p) == 1000000)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all combinatorial_rank tests passed")
