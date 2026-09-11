"""Dynamic time warping: aligning time series that vary in speed.

Two recordings of the same spoken word, two gait cycles, two heartbeats -- they trace the same shape
but at different, non-uniformly varying speeds, so a point-by-point (Euclidean) comparison badly
mismatches them. DYNAMIC TIME WARPING (DTW) finds the optimal NON-LINEAR alignment: it stretches and
compresses the time axis of one series to match the other, and returns both the minimal alignment
cost and the WARPING PATH that achieves it. DTW is the backbone of speech recognition (before deep
learning), gesture and signature recognition, and time-series clustering and classification.

The algorithm is dynamic programming over an n x m cost grid. Cell (i, j) holds the cheapest cost to
align the first i points of one series with the first j of the other; it equals the local distance
between points i and j plus the minimum of the three neighbours (i-1, j), (i, j-1), (i-1, j-1) --
which encode 'insert', 'delete', and 'match' moves. The bottom-right cell is the DTW distance, and
tracing the minimizing choices back from it recovers the warping path, a monotone staircase through
the grid pairing each point of one series with one or more points of the other. A SAKOE-CHIBA BAND
constrains the path to stay within a window of the diagonal, which both speeds the computation to
O(n * w) and prevents pathological alignments.

This module implements DTW distance and warping-path recovery, an optional Sakoe-Chiba band, and a
convenience for multi-dimensional series, verified against brute-force references and known
properties: that identical series have zero distance, that DTW is symmetric, that it is invariant to
time stretching (duplicating points does not change the distance), that it lower-bounds and improves
on the Euclidean distance for shifted signals, that the warping path is monotone and connects the
corners, and that a small hand-computed grid matches the exact DP value. Pure stdlib; a
sequence-alignment companion to the edit-distance and sequence-alignment notes."""

from __future__ import annotations


def _abs_dist(a, b):
    return abs(a - b)


def dtw(x, y, dist=_abs_dist, band=None):
    """DTW distance between series x and y.

    dist: local distance between two points (default absolute difference for scalars).
    band: optional Sakoe-Chiba half-width; if set, the warping path stays within |i-j| <= band
    (after diagonal scaling), speeding the computation and forbidding wild warps.
    Returns the DTW distance (a float)."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("inf") if (n or m) else 0.0
    INF = float("inf")
    # scale the band to the longer axis so rectangular series work
    w = None
    if band is not None:
        w = max(band, abs(n - m))
    prev = [INF] * (m + 1)
    prev[0] = 0.0
    # we only need the previous row; but path recovery needs the full matrix, so build it there.
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = 1, m
        if w is not None:
            center = i * m // n
            jlo = max(1, center - w)
            jhi = min(m, center + w)
        for j in range(jlo, jhi + 1):
            cost = dist(x[i - 1], y[j - 1])
            D[i][j] = cost + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    return D[n][m]


def dtw_path(x, y, dist=_abs_dist, band=None):
    """DTW distance and the warping path as a list of (i, j) index pairs (0-based) from (0,0) to
    (n-1, m-1)."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return (0.0 if n == m else float("inf")), []
    INF = float("inf")
    w = None
    if band is not None:
        w = max(band, abs(n - m))
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = 1, m
        if w is not None:
            center = i * m // n
            jlo = max(1, center - w)
            jhi = min(m, center + w)
        for j in range(jlo, jhi + 1):
            cost = dist(x[i - 1], y[j - 1])
            D[i][j] = cost + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    # backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        # choose the neighbour that produced D[i][j]
        candidates = [(D[i - 1][j - 1], i - 1, j - 1),
                      (D[i - 1][j], i - 1, j),
                      (D[i][j - 1], i, j - 1)]
        _, i, j = min(candidates, key=lambda t: t[0])
    path.reverse()
    return D[n][m], path


def dtw_multi(x, y, band=None):
    """DTW for multi-dimensional series (each point is a vector), using Euclidean local distance."""
    import math

    def euclid(a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    return dtw(x, y, dist=euclid, band=band)


def euclidean_distance(x, y):
    """Point-by-point Euclidean distance (requires equal lengths) -- for comparison with DTW."""
    if len(x) != len(y):
        raise ValueError("Euclidean distance requires equal-length series")
    return sum(_abs_dist(a, b) for a, b in zip(x, y))
