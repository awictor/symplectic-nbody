"""Tests for sequence_alignment: global/local DP, traceback, scoring, gaps, identity."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sequence_alignment import (needleman_wunsch, smith_waterman, alignment_score, identity)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- global alignment: aligned strings are equal length --------------------
score, aa, bb = needleman_wunsch("GCATGCU", "GATTACA")
check("aligned strings have equal length", len(aa) == len(bb))
check("global score matches recomputation", alignment_score(aa, bb) == score)
check("removing gaps recovers the originals",
      aa.replace("-", "") == "GCATGCU" and bb.replace("-", "") == "GATTACA")

# --- identical sequences align perfectly -----------------------------------
s_id, a_id, b_id = needleman_wunsch("HELLO", "HELLO")
check("identical sequences score = length * match", s_id == 5)
check("identical alignment has no gaps", "-" not in a_id and "-" not in b_id)
check("identical alignment is the sequence", a_id == "HELLO" and b_id == "HELLO")

# --- a pure deletion inserts one gap ---------------------------------------
sd, ad, bd = needleman_wunsch("ACGT", "AGT")
check("deletion aligns with a single gap", bd.count("-") == 1)
check("deletion alignment recovers sequences",
      ad.replace("-", "") == "ACGT" and bd.replace("-", "") == "AGT")
check("deletion score is consistent", alignment_score(ad, bd) == sd)

# --- a pure insertion --------------------------------------------------------
si, ai, bi = needleman_wunsch("AGT", "ACGT")
check("insertion aligns with a single gap", ai.count("-") == 1)

# --- global score by hand: 'AAAA' vs 'AAAA' with match=2 -------------------
s2, _, _ = needleman_wunsch("AAAA", "AAAA", match=2)
check("custom match score", s2 == 8)
# a single mismatch costs match - (match - mismatch)
sm, _, _ = needleman_wunsch("AAA", "ABA", match=1, mismatch=-1)
check("one mismatch in the middle", sm == 1)   # A match + B/A mismatch + A match = 1-1+1

# --- gap penalty influences the alignment ----------------------------------
# with a huge gap penalty, mismatches are preferred over gaps
s_hi, a_hi, b_hi = needleman_wunsch("AAAA", "ATAA", gap=-10)
check("high gap penalty avoids gaps", "-" not in a_hi and "-" not in b_hi)
# with a small gap penalty vs large mismatch, gaps may be preferred
s_lo, a_lo, b_lo = needleman_wunsch("AAAA", "ATAA", mismatch=-10, gap=-1)
check("cheap gaps preferred over costly mismatch", "-" in a_lo or "-" in b_lo)

# --- Smith-Waterman finds an embedded motif --------------------------------
sl, al, bl = smith_waterman("xxxGATTACAyyy", "zzGATTACAww")
check("local alignment finds the shared motif", al == "GATTACA" and bl == "GATTACA")
check("local score is positive", sl > 0)
check("local score = motif length * match", sl == 7 * 2)

# --- local alignment ignores flanking mismatches ---------------------------
sl2, al2, bl2 = smith_waterman("AAAAHELLOBBBB", "CCCCHELLODDDD")
check("local finds HELLO despite different flanks", al2 == "HELLO" and bl2 == "HELLO")

# --- local alignment of totally dissimilar sequences is short/zero ----------
sl3, al3, bl3 = smith_waterman("ABCDEF", "UVWXYZ")
check("no common subsequence -> score 0", sl3 == 0)

# --- global vs local on the same embedded motif ----------------------------
# global drags in the mismatched flanks; local isolates the motif
g_score, _, _ = needleman_wunsch("AAHELLOAA", "BBHELLOBB")
l_score, la, lb = smith_waterman("AAHELLOAA", "BBHELLOBB")
check("local isolates the motif", la == "HELLO" and lb == "HELLO")
check("local score exceeds the global score here", l_score > g_score)

# --- identity ---------------------------------------------------------------
check("identity of identical strings is 1", identity("HELLO", "HELLO") == 1.0)
check("identity with one mismatch", abs(identity("HELLO", "HELLX") - 0.8) < 1e-9)
check("identity ignores gap columns", identity("AC-GT", "ACAGT") == 1.0)

# --- empty inputs -----------------------------------------------------------
se, ae, be = needleman_wunsch("", "ABC")
check("empty vs non-empty is all gaps", ae == "---" and be == "ABC")
check("empty-empty alignment", needleman_wunsch("", "") == (0, "", ""))
check("local empty is score 0", smith_waterman("", "ABC") == (0, "", ""))

# --- symmetry of the global score ------------------------------------------
s_ab, _, _ = needleman_wunsch("KITTEN", "SITTING")
s_ba, _, _ = needleman_wunsch("SITTING", "KITTEN")
check("global score is symmetric", s_ab == s_ba)

# --- traceback is always consistent with the reported score ----------------
for x, y in [("KITTEN", "SITTING"), ("ACGTACGT", "ACTACT"), ("banana", "bandana")]:
    sc, xa, ya = needleman_wunsch(x, y)
    check(f"score consistent for {x}/{y}", alignment_score(xa, ya) == sc)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all sequence_alignment tests passed")
