"""Tests for hirschberg: linear-space alignment vs full-matrix Needleman-Wunsch."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hirschberg import (align, alignment_score, lcs, lcs_length, full_alignment, GAP)
from lcs import lcs_length as ref_lcs_length

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


def _is_subseq(sub, seq):
    it = iter(seq)
    return all(c in it for c in sub)


def valid_alignment(a, b, aa, bb, match=1, mismatch=-1, gap=-1):
    """Alignment is same length, degaps to originals, has no gap-vs-gap column, and its score is the
    given value."""
    if len(aa) != len(bb):
        return None
    if aa.replace(GAP, "") != a or bb.replace(GAP, "") != b:
        return None
    score = 0
    for x, y in zip(aa, bb):
        if x == GAP and y == GAP:
            return None
        if x == GAP or y == GAP:
            score += gap
        elif x == y:
            score += match
        else:
            score += mismatch
    return score


# --- known case -------------------------------------------------------------
a, b = "AGTACGCA", "TATGC"
aa, bb = align(a, b)
sc, _, _ = full_alignment(a, b)
check("known pair: Hirschberg score equals full NW", alignment_score(a, b) == sc)
check("known pair: alignment degaps to the originals",
      aa.replace(GAP, "") == a and bb.replace(GAP, "") == b)

check("empty vs empty aligns to empty", align("", "") == ("", ""))
check("empty vs string is all gaps", align("", "ABC") == ("---", "ABC"))
check("string vs empty is all gaps", align("AB", "") == ("AB", "--"))
check("identical strings align with no gaps", align("HELLO", "HELLO") == ("HELLO", "HELLO"))

# --- score matches full NW on random pairs ---------------------------------
rng = LCG(2026)
score_ok = align_ok = True
for _ in range(500):
    n1 = rng.randint(0, 14)
    n2 = rng.randint(0, 14)
    a = random_str(rng, n1, "ACGT")
    b = random_str(rng, n2, "ACGT")
    hs = alignment_score(a, b)
    fs, _, _ = full_alignment(a, b)
    if hs != fs:
        score_ok = False
        print(f"  score mismatch: hirsch={hs} full={fs} a={a!r} b={b!r}")
        break
    aa, bb = align(a, b)
    got = valid_alignment(a, b, aa, bb)
    if got is None or got != fs:
        align_ok = False
        print(f"  bad alignment: a={a!r} b={b!r} aa={aa!r} bb={bb!r} score={got} want={fs}")
        break
check("Hirschberg score equals full Needleman-Wunsch (500 random pairs)", score_ok)
check("Hirschberg alignment is valid and achieves the optimal score", align_ok)

# --- different scoring schemes ---------------------------------------------
rng = LCG(4242)
scheme_ok = True
for _ in range(200):
    a = random_str(rng, rng.randint(0, 12), "AB")
    b = random_str(rng, rng.randint(0, 12), "AB")
    for (mt, mm, gp) in [(2, -1, -2), (1, -1, -1), (3, -2, -1)]:
        hs = alignment_score(a, b, mt, mm, gp)
        fs, _, _ = full_alignment(a, b, mt, mm, gp)
        if hs != fs:
            scheme_ok = False
            break
        aa, bb = align(a, b, mt, mm, gp)
        if valid_alignment(a, b, aa, bb, mt, mm, gp) != fs:
            scheme_ok = False
            break
    if not scheme_ok:
        break
check("Hirschberg matches full NW under several scoring schemes", scheme_ok)

# --- LCS as a special case matches the reference DP ------------------------
rng = LCG(777)
lcs_ok = True
for _ in range(400):
    a = random_str(rng, rng.randint(0, 14), "ACGT")
    b = random_str(rng, rng.randint(0, 14), "ACGT")
    if lcs_length(a, b) != ref_lcs_length(a, b):
        lcs_ok = False
        print(f"  lcs length mismatch: a={a!r} b={b!r}")
        break
    got = lcs(a, b)
    # the returned LCS must be a subsequence of both and have the right length
    if len(got) != ref_lcs_length(a, b):
        lcs_ok = False
        break
    if not (_is_subseq(got, a) and _is_subseq(got, b)):
        lcs_ok = False
        break
check("Hirschberg LCS length and witness match the reference DP (400 pairs)", lcs_ok)

# --- long pair (memory-prohibitive for the full matrix elsewhere) ----------
rng = LCG(31337)
a = random_str(rng, 3000, "ACGT")
b = random_str(rng, 3000, "ACGT")
aa, bb = align(a, b)
check("3000x3000 alignment: valid and degaps to originals",
      aa.replace(GAP, "") == a and bb.replace(GAP, "") == b)
check("3000x3000 alignment: Hirschberg score is self-consistent",
      valid_alignment(a, b, aa, bb) == alignment_score(a, b))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all hirschberg tests passed")
