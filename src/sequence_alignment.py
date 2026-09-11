"""Sequence alignment: Needleman-Wunsch (global) and Smith-Waterman (local).

Aligning two sequences -- lining them up to maximize matching symbols by inserting gaps -- is the
central operation of bioinformatics (comparing DNA, proteins), spell-check, diff tools, and plagiarism
detection. Where edit distance just counts operations, alignment produces the actual correspondence
and a SCORE, weighting matches, mismatches, and gaps however the problem demands.

Both algorithms fill a dynamic-programming matrix where cell (i, j) is the best score aligning the
first i symbols of one sequence with the first j of the other, from three moves:

    diagonal -> align a[i] with b[j]   (+match or -mismatch score)
    up       -> a gap in b             (-gap penalty)
    left     -> a gap in a             (-gap penalty)

They differ in the boundary and the objective:

  NEEDLEMAN-WUNSCH (global): align the sequences end to end; the first row/column are cumulative
      gap penalties, and the answer is the bottom-right cell. Best for comparing whole sequences of
      similar length.
  SMITH-WATERMAN (local): find the best-matching SUBSEQUENCES; scores are floored at zero (a bad
      stretch resets rather than dragging the alignment down), and the answer starts from the
      matrix's maximum cell, tracing back until it hits a zero. Best for finding a conserved motif
      inside larger, dissimilar sequences.

TRACEBACK walks the matrix backwards along the chosen moves to reconstruct the two gapped, aligned
strings. This module implements both with a configurable match/mismatch/gap scoring, returning the
score and the aligned strings -- verified that identical sequences align perfectly with the maximum
score, that the global score matches recomputing it from the alignment, that a local alignment finds
an embedded motif, that gap penalties insert gaps where expected, and against hand-worked examples.
Pure stdlib; a dynamic-programming companion to the Levenshtein edit-distance note (which counts
edits; this one scores and reconstructs the alignment)."""

from __future__ import annotations

_GAP = "-"


def needleman_wunsch(a, b, match=1, mismatch=-1, gap=-1):
    """Global alignment of sequences a and b. Returns (score, aligned_a, aligned_b)."""
    n, m = len(a), len(b)
    # score matrix; first row/col are cumulative gap penalties
    H = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        H[i][0] = i * gap
    for j in range(1, m + 1):
        H[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            H[i][j] = max(H[i - 1][j - 1] + s,      # diagonal (align)
                          H[i - 1][j] + gap,        # up (gap in b)
                          H[i][j - 1] + gap)        # left (gap in a)
    # traceback from the bottom-right corner
    ai, bi = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match if a[i - 1] == b[j - 1] else mismatch
            if H[i][j] == H[i - 1][j - 1] + s:
                ai.append(a[i - 1]); bi.append(b[j - 1]); i -= 1; j -= 1; continue
        if i > 0 and H[i][j] == H[i - 1][j] + gap:
            ai.append(a[i - 1]); bi.append(_GAP); i -= 1; continue
        # else left (gap in a)
        ai.append(_GAP); bi.append(b[j - 1]); j -= 1
    return H[n][m], "".join(reversed(ai)), "".join(reversed(bi))


def smith_waterman(a, b, match=2, mismatch=-1, gap=-1):
    """Local alignment: the best-scoring pair of subsequences. Returns (score, aligned_a,
    aligned_b) for the highest-scoring local region."""
    n, m = len(a), len(b)
    H = [[0] * (m + 1) for _ in range(n + 1)]      # first row/col stay 0 (local: can start anywhere)
    best = 0
    best_pos = (0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            H[i][j] = max(0,                        # floor at zero: a bad stretch resets
                          H[i - 1][j - 1] + s,
                          H[i - 1][j] + gap,
                          H[i][j - 1] + gap)
            if H[i][j] > best:
                best = H[i][j]
                best_pos = (i, j)
    # traceback from the maximum cell until a zero is reached
    ai, bi = [], []
    i, j = best_pos
    while i > 0 and j > 0 and H[i][j] > 0:
        s = match if a[i - 1] == b[j - 1] else mismatch
        if H[i][j] == H[i - 1][j - 1] + s:
            ai.append(a[i - 1]); bi.append(b[j - 1]); i -= 1; j -= 1
        elif H[i][j] == H[i - 1][j] + gap:
            ai.append(a[i - 1]); bi.append(_GAP); i -= 1
        else:
            ai.append(_GAP); bi.append(b[j - 1]); j -= 1
    return best, "".join(reversed(ai)), "".join(reversed(bi))


def alignment_score(aligned_a, aligned_b, match=1, mismatch=-1, gap=-1):
    """Recompute the score of a pair of aligned (gapped) strings -- a check on the DP result."""
    total = 0
    for ca, cb in zip(aligned_a, aligned_b):
        if ca == _GAP or cb == _GAP:
            total += gap
        elif ca == cb:
            total += match
        else:
            total += mismatch
    return total


def identity(aligned_a, aligned_b):
    """Fraction of aligned columns that are identical (excluding gap columns)."""
    matches = cols = 0
    for ca, cb in zip(aligned_a, aligned_b):
        if ca == _GAP or cb == _GAP:
            continue
        cols += 1
        if ca == cb:
            matches += 1
    return matches / cols if cols else 0.0
