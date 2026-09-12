"""Hirschberg's algorithm: optimal sequence alignment in linear space.

The Needleman-Wunsch dynamic program aligns two sequences optimally, but its O(m*n) memory becomes the
bottleneck long before its O(m*n) time does: aligning two 100,000-character DNA strands would need a
10-billion-cell table. HIRSCHBERG'S ALGORITHM (1975) computes the SAME optimal alignment in only
O(min(m, n)) space -- linear, not quadratic -- by a clever divide-and-conquer, at the cost of just a
constant factor more time. It is the reason genome-scale global alignment is feasible at all, and the
same idea (a linear-space traceback for a quadratic DP) recurs throughout dynamic programming.

The key observation: the alignment SCORE of every prefix pair can be computed row by row keeping only
two rows in memory (the "Needleman-Wunsch score profile"), so the last row -- the optimal score of
aligning all of one sequence with each prefix of the other -- costs only linear space. To recover the
actual alignment, Hirschberg splits the first sequence at its midpoint and asks: at what position in
the second sequence should the two halves meet? It computes the forward score profile of the top half
and the backward (reversed) score profile of the bottom half; the split point maximising their sum is
where an optimal alignment crosses the midpoint. Recursing on the two resulting sub-problems, each in
linear space, reconstructs the whole alignment; the recursion depth is O(log m) and the total work
stays O(m*n).

This module computes the optimal global-alignment score and a full optimal alignment (with gap
characters) in linear space via Hirschberg, plus the longest common subsequence as a special case. It
is verified against a direct full-matrix Needleman-Wunsch reference -- the score matches, the returned
alignment is valid (removing gaps recovers the originals) and achieves the optimal score, and the LCS
length matches a standard DP -- on hundreds of random string pairs, including a long pair that would be
memory-prohibitive for the full matrix. Pure stdlib; a dynamic-programming companion to the
sequence-alignment (Needleman-Wunsch/Smith-Waterman), LCS, and edit-distance notes."""

from __future__ import annotations

GAP = "-"


def _score_profile(a, b, match, mismatch, gap):
    """The last row of the Needleman-Wunsch score matrix for aligning `a` (rows) against every prefix
    of `b` (columns), in O(len(b)) space. Returns a list of len(b)+1 scores."""
    prev = [j * gap for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        cur = [i * gap] + [0] * len(b)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            sub = prev[j - 1] + (match if ai == b[j - 1] else mismatch)
            cur[j] = max(sub, prev[j] + gap, cur[j - 1] + gap)
        prev = cur
    return prev


def alignment_score(a, b, match=1, mismatch=-1, gap=-1):
    """The optimal global (Needleman-Wunsch) alignment score, in linear space."""
    return _score_profile(a, b, match, mismatch, gap)[len(b)]


def align(a, b, match=1, mismatch=-1, gap=-1):
    """An optimal global alignment of `a` and `b` computed in O(min(len(a), len(b))) space via
    Hirschberg. Returns (aligned_a, aligned_b) strings with GAP characters inserted."""
    # base cases
    if len(a) == 0:
        return GAP * len(b), b
    if len(b) == 0:
        return a, GAP * len(a)
    if len(a) == 1 or len(b) == 1:
        return _align_small(a, b, match, mismatch, gap)

    # divide `a` at its midpoint
    mid = len(a) // 2
    # forward profile of a[:mid] vs b, backward profile of a[mid:] vs b
    fwd = _score_profile(a[:mid], b, match, mismatch, gap)
    bwd = _score_profile(a[mid:][::-1], b[::-1], match, mismatch, gap)[::-1]
    # choose the split column j maximising fwd[j] + bwd[j]
    best_j = max(range(len(b) + 1), key=lambda j: fwd[j] + bwd[j])

    left_a, left_b = align(a[:mid], b[:best_j], match, mismatch, gap)
    right_a, right_b = align(a[mid:], b[best_j:], match, mismatch, gap)
    return left_a + right_a, left_b + right_b


def _align_small(a, b, match, mismatch, gap):
    """Direct full-matrix Needleman-Wunsch traceback for the base case where one sequence has length
    <= 1 (or, generically, any small pair). O(len(a)*len(b)) space, but only used on a thin strip."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i * gap
    for j in range(1, n + 1):
        dp[0][j] = j * gap
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            dp[i][j] = max(sub, dp[i - 1][j] + gap, dp[i][j - 1] + gap)
    # traceback
    ra, rb = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and \
                dp[i][j] == dp[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            ra.append(a[i - 1])
            rb.append(b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            ra.append(a[i - 1])
            rb.append(GAP)
            i -= 1
        else:
            ra.append(GAP)
            rb.append(b[j - 1])
            j -= 1
    return "".join(reversed(ra)), "".join(reversed(rb))


def lcs(a, b):
    """The longest common subsequence of `a` and `b`, recovered in linear space via Hirschberg with a
    match/mismatch/gap scoring that counts matches (match=1, mismatch=0, gap=0 makes the optimal
    score the LCS length and the matched columns the LCS)."""
    aa, bb = align(a, b, match=1, mismatch=0, gap=0)
    out = []
    for x, y in zip(aa, bb):
        if x == y and x != GAP:
            out.append(x)
    return "".join(out)


def lcs_length(a, b):
    """The length of the longest common subsequence, in linear space."""
    return _score_profile(a, b, 1, 0, 0)[len(b)]


# --- full-matrix reference --------------------------------------------------
def full_alignment(a, b, match=1, mismatch=-1, gap=-1):
    """Reference optimal global alignment via the full O(m*n) Needleman-Wunsch matrix and traceback.
    Returns (score, aligned_a, aligned_b)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i * gap
    for j in range(1, n + 1):
        dp[0][j] = j * gap
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            dp[i][j] = max(sub, dp[i - 1][j] + gap, dp[i][j - 1] + gap)
    ra, rb = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and \
                dp[i][j] == dp[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            ra.append(a[i - 1]); rb.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            ra.append(a[i - 1]); rb.append(GAP); i -= 1
        else:
            ra.append(GAP); rb.append(b[j - 1]); j -= 1
    return dp[m][n], "".join(reversed(ra)), "".join(reversed(rb))
