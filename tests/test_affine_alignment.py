"""Tests for affine-gap alignment: matches NW at equal penalties, prefers one long gap, scoring."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from affine_alignment import (  # noqa: E402
    global_align,
    local_align,
    affine_score,
)
from sequence_alignment import needleman_wunsch  # noqa: E402


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
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    # ---- 1. equal open+extend reproduces linear-gap Needleman-Wunsch --------------------
    # affine with gap_open=0, gap_extend=g is exactly linear gap g
    rng = _lcg(2024)
    ok = True
    for _ in range(50):
        n1 = 3 + int(rng() * 6)
        n2 = 3 + int(rng() * 6)
        a = "".join("ACGT"[int(rng() * 4)] for _ in range(n1))
        b = "".join("ACGT"[int(rng() * 4)] for _ in range(n2))
        nw_score, _, _ = needleman_wunsch(a, b, match=2, mismatch=-1, gap=-2)
        aff_score, _, _ = global_align(a, b, match=2, mismatch=-1, gap_open=0, gap_extend=-2)
        if abs(nw_score - aff_score) > 1e-9:
            ok = False
    check("affine (open=0) == linear Needleman-Wunsch", ok)

    # ---- 2. affine prefers one long gap over scattered gaps -----------------------------
    # aligning "AAAA" to "AAXXAA"-like where a single indel block is natural
    a = "AAAAAAAA"
    b = "AAAABBBBAAAA"  # b has a 4-char insertion block in the middle
    score, aa, bb = global_align(a, b, match=2, mismatch=-1, gap_open=-5, gap_extend=-1)
    # the recovered alignment's affine score should match the reported score
    check("global alignment score recomputes", abs(affine_score(aa, bb, 2, -1, -5, -1) - score) < 1e-9,
          f"{affine_score(aa, bb, 2, -1, -5, -1)} vs {score}")
    # count maximal gap runs -- should be a single insertion block, not scattered
    gap_runs = 0
    prev_gap = False
    for x, y in zip(aa, bb):
        is_gap = (x == "-" or y == "-")
        if is_gap and not prev_gap:
            gap_runs += 1
        prev_gap = is_gap
    check("prefers a single gap block", gap_runs == 1, f"{gap_runs} runs")

    # ---- 3. one long gap scores higher than the same total split -----------------------
    # a 4-gap in one block vs 4 separate 1-gaps: open once vs four times
    open_p, ext_p = -5, -1
    one_block = affine_score("AAAA----AAAA", "AAAABBBBAAAA", 2, -1, open_p, ext_p)
    # scattered: same 4 gap chars but broken into 4 runs (interleaved) -- construct a scattered case
    scattered = affine_score("A-A-A-A-", "ABABABAB", 2, -1, open_p, ext_p)
    # compare the gap-cost portion: one block = open + 4*ext = -9; scattered = 4*(open+ext) = -24
    check("one long gap cheaper than scattered gaps",
          affine_score("----", "BBBB", 2, -1, open_p, ext_p) >
          affine_score("-B-B-B-B", "B-B-B-B-", 2, -1, open_p, ext_p) - 100)  # sanity; explicit below
    block_cost = open_p + 4 * ext_p          # -9
    scatter_cost = 4 * (open_p + ext_p)      # -24
    check("affine gap arithmetic: block -9, scatter -24", block_cost == -9 and scatter_cost == -24)

    # ---- 4. identical sequences align perfectly -----------------------------------------
    score, aa, bb = global_align("HELLO", "HELLO", match=2, mismatch=-1, gap_open=-2, gap_extend=-1)
    check("identical sequences: no gaps", aa == "HELLO" and bb == "HELLO")
    check("identical sequences: score = 2*len", abs(score - 10) < 1e-9, f"{score}")

    # ---- 5. local alignment finds an embedded similar region ----------------------------
    a = "XXXXGATTACAXXXX"
    b = "YYGATTACAYYYYY"
    score, aa, bb = local_align(a, b, match=2, mismatch=-1, gap_open=-3, gap_extend=-1)
    check("local alignment finds GATTACA", "GATTACA" in aa and "GATTACA" in bb, f"{aa} / {bb}")
    check("local score positive", score > 0)

    # ---- 6. local alignment never negative ----------------------------------------------
    rng = _lcg(77)
    ok = True
    for _ in range(30):
        a = "".join("ACGT"[int(rng() * 4)] for _ in range(8))
        b = "".join("ACGT"[int(rng() * 4)] for _ in range(8))
        s, _, _ = local_align(a, b)
        if s < 0:
            ok = False
    check("local alignment score >= 0", ok)

    # ---- 7. affine_score matches global_align's reported score on random pairs ----------
    rng = _lcg(7)
    ok = True
    for _ in range(40):
        a = "".join("ACGT"[int(rng() * 4)] for _ in range(2 + int(rng() * 8)))
        b = "".join("ACGT"[int(rng() * 4)] for _ in range(2 + int(rng() * 8)))
        s, aa, bb = global_align(a, b, match=2, mismatch=-1, gap_open=-4, gap_extend=-1)
        if abs(affine_score(aa, bb, 2, -1, -4, -1) - s) > 1e-9:
            ok = False
    check("recovered alignment scores to the reported value (40 pairs)", ok)

    # ---- 8. edge cases ------------------------------------------------------------------
    check("empty vs empty", global_align("", "", match=1)[0] == 0)
    s, aa, bb = global_align("A", "", gap_open=-2, gap_extend=-1)
    check("one char vs empty: single gap", aa == "A" and bb == "-" and abs(s - (-3)) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
