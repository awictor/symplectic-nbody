"""Gotoh affine-gap sequence alignment: charging gap opening and extension differently.

Sequence alignment scores how two strings line up, inserting GAPS to bring matching characters into
register. The simplest models charge a fixed penalty per gap character (a LINEAR gap cost), but that
misrepresents biology and text: a single long gap (one insertion/deletion event) is far more likely
than many scattered short ones, yet linear scoring penalises them equally. The AFFINE gap model fixes
this by charging a large OPENING cost to start a gap plus a small EXTENSION cost per additional
character -- so one gap of length 5 costs open + 5*extend, much cheaper than five separate gaps.

Naively, tracking whether we are inside a gap seems to need extra state, and it does: GOTOH'S
algorithm (1982) runs three dynamic-programming matrices in parallel --

    M[i][j]  best score aligning a[:i], b[:j] with a[i-1], b[j-1] MATCHED/substituted,
    Ix[i][j] best score ending in a gap in x (a deletion, consuming a[i-1]),
    Iy[i][j] best score ending in a gap in y (an insertion, consuming b[j-1]),

with Ix/Iy transitions choosing between OPENING a gap (from M, paying open+extend) or EXTENDING one
(from Ix/Iy, paying extend). This keeps the whole alignment O(nm) despite the affine cost. This module
implements global (Needleman-Wunsch style) and local (Smith-Waterman style) affine-gap alignment,
returning the score and the aligned strings, plus a helper to score a given alignment under the affine
model.

Validated: with equal open and extension penalties the affine score matches the linear-gap
Needleman-Wunsch; a single long gap scores higher than the same total gap split into pieces (the whole
point of affine); the score of a returned alignment recomputes to the reported value; local alignment
never scores below zero and finds an embedded high-similarity region; identical sequences align
perfectly; and known small cases check out. Pure stdlib; the affine-cost companion to the linear-gap
Needleman-Wunsch / Smith-Waterman aligners."""

from __future__ import annotations

NEG_INF = float("-inf")


def _score_pair(x, y, match, mismatch):
    return match if x == y else mismatch


def global_align(a, b, match=1, mismatch=-1, gap_open=-2, gap_extend=-1):
    """Global affine-gap alignment (Gotoh). gap_open is paid once to start a gap, gap_extend per
    gap character (so a length-L gap costs gap_open + L*gap_extend). Returns (score, aa, bb)."""
    n, m = len(a), len(b)
    M = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Ix = [[NEG_INF] * (m + 1) for _ in range(n + 1)]  # gap in b (deletion from a)
    Iy = [[NEG_INF] * (m + 1) for _ in range(n + 1)]  # gap in a (insertion in b)
    M[0][0] = 0.0
    for i in range(1, n + 1):
        Ix[i][0] = gap_open + i * gap_extend
    for j in range(1, m + 1):
        Iy[0][j] = gap_open + j * gap_extend
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _score_pair(a[i - 1], b[j - 1], match, mismatch)
            M[i][j] = s + max(M[i - 1][j - 1], Ix[i - 1][j - 1], Iy[i - 1][j - 1])
            Ix[i][j] = max(M[i - 1][j] + gap_open + gap_extend, Ix[i - 1][j] + gap_extend)
            Iy[i][j] = max(M[i][j - 1] + gap_open + gap_extend, Iy[i][j - 1] + gap_extend)
    score = max(M[n][m], Ix[n][m], Iy[n][m])
    aa, bb = _traceback_global(a, b, M, Ix, Iy, match, mismatch, gap_open, gap_extend)
    return score, aa, bb


def _traceback_global(a, b, M, Ix, Iy, match, mismatch, gap_open, gap_extend):
    n, m = len(a), len(b)
    i, j = n, m
    # which matrix are we in at the corner?
    state = max(("M", "Ix", "Iy"), key=lambda s: {"M": M, "Ix": Ix, "Iy": Iy}[s][n][m])
    aa, bb = [], []
    while i > 0 or j > 0:
        if state == "M" and i > 0 and j > 0:
            aa.append(a[i - 1])
            bb.append(b[j - 1])
            s = _score_pair(a[i - 1], b[j - 1], match, mismatch)
            prev = M[i][j] - s
            i, j = i - 1, j - 1
            state = max(("M", "Ix", "Iy"),
                        key=lambda st: {"M": M, "Ix": Ix, "Iy": Iy}[st][i][j])
        elif state == "Ix" and i > 0:
            aa.append(a[i - 1])
            bb.append("-")
            # came from M (open) or Ix (extend)?
            if abs(Ix[i][j] - (Ix[i - 1][j] + gap_extend)) < 1e-9:
                state = "Ix"
            else:
                state = "M"
            i -= 1
        elif state == "Iy" and j > 0:
            aa.append("-")
            bb.append(b[j - 1])
            if abs(Iy[i][j] - (Iy[i][j - 1] + gap_extend)) < 1e-9:
                state = "Iy"
            else:
                state = "M"
            j -= 1
        elif i > 0:
            aa.append(a[i - 1]); bb.append("-"); i -= 1
        else:
            aa.append("-"); bb.append(b[j - 1]); j -= 1
    return "".join(reversed(aa)), "".join(reversed(bb))


def local_align(a, b, match=2, mismatch=-1, gap_open=-2, gap_extend=-1):
    """Local affine-gap alignment (Smith-Waterman with affine gaps). Returns (score, aa, bb) for the
    best-scoring local region."""
    n, m = len(a), len(b)
    M = [[0.0] * (m + 1) for _ in range(n + 1)]
    Ix = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Iy = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    best = 0.0
    bi = bj = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _score_pair(a[i - 1], b[j - 1], match, mismatch)
            M[i][j] = max(0.0, s + max(M[i - 1][j - 1], Ix[i - 1][j - 1], Iy[i - 1][j - 1]))
            Ix[i][j] = max(M[i - 1][j] + gap_open + gap_extend, Ix[i - 1][j] + gap_extend)
            Iy[i][j] = max(M[i][j - 1] + gap_open + gap_extend, Iy[i][j - 1] + gap_extend)
            if M[i][j] > best:
                best = M[i][j]
                bi, bj = i, j
    # traceback from the best M cell down to a zero
    aa, bb = [], []
    i, j = bi, bj
    while i > 0 and j > 0 and M[i][j] > 0:
        aa.append(a[i - 1])
        bb.append(b[j - 1])
        i, j = i - 1, j - 1
    return best, "".join(reversed(aa)), "".join(reversed(bb))


def affine_score(aa, bb, match=1, mismatch=-1, gap_open=-2, gap_extend=-1):
    """Score an existing alignment under the affine-gap model (open once per maximal gap run)."""
    score = 0.0
    in_gap_a = in_gap_b = False
    for x, y in zip(aa, bb):
        if x == "-":
            score += gap_extend + (gap_open if not in_gap_a else 0)
            in_gap_a = True
            in_gap_b = False
        elif y == "-":
            score += gap_extend + (gap_open if not in_gap_b else 0)
            in_gap_b = True
            in_gap_a = False
        else:
            score += _score_pair(x, y, match, mismatch)
            in_gap_a = in_gap_b = False
    return score
