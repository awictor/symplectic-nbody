"""Soft-DTW: a smooth, differentiable version of dynamic time warping.

Dynamic time warping aligns two time series by finding the lowest-cost monotonic correspondence between
their points, absorbing shifts and speed differences that break ordinary Euclidean distance. Its cost
is a MINIMUM over exponentially many alignment paths, computed by a min-plus dynamic program. That min
makes DTW non-differentiable and jagged: an infinitesimal change to a series can flip which path is
optimal, so the DTW distance has kinks and cannot be used as a smooth loss for gradient-based learning
or for averaging a set of series.

Cuturi and Blondel's SOFT-DTW (2017) replaces the hard minimum with a SOFT MINIMUM -- a log-sum-exp with
a temperature gamma:  softmin_gamma(a_1, ..., a_k) = -gamma * log sum_i exp(-a_i / gamma). As gamma -> 0
the softmin becomes the ordinary minimum and soft-DTW recovers classic DTW; as gamma grows it averages
over ALL alignment paths, weighting each by exp(-cost/gamma). The result is smooth and differentiable
everywhere, which is exactly what you need to use alignment as a differentiable loss, to compute a
soft-DTW BARYCENTER (a representative average of several series under warping), or to backpropagate
through time-series matching.

The recurrence is the same dynamic program as DTW with the min swapped for the softmin:
r[i][j] = D[i][j] + softmin( r[i-1][j], r[i][j-1], r[i-1][j-1] ), where D is the pairwise squared-cost
matrix. This module computes soft-DTW, the classic (hard) DTW for comparison, and the soft alignment
matrix E[i][j] = the probability that cell (i,j) lies on an alignment path (the gradient of soft-DTW
with respect to the cost matrix, from the Cuturi-Blondel backward recursion). It is validated: soft-DTW
is symmetric; soft-DTW of a series with itself is <= that of two different series; as gamma -> 0 it
converges to hard DTW computed independently; softmin is bounded between the true min and the min minus
gamma*log(k); the alignment matrix is a valid distribution that concentrates on the optimal path as
gamma -> 0; and it is smooth (a small perturbation changes it by a small amount, unlike hard DTW).
Pure stdlib; the differentiable companion to the DTW and matrix-profile tools."""

from __future__ import annotations

import math


def _softmin(a, b, c, gamma):
    """Soft minimum of three values with temperature gamma (numerically stable)."""
    m = min(a, b, c)
    if gamma <= 0:
        return m
    s = (math.exp(-(a - m) / gamma) + math.exp(-(b - m) / gamma) + math.exp(-(c - m) / gamma))
    return m - gamma * math.log(s)


def _cost_matrix(x, y):
    """Pairwise squared-Euclidean cost between scalar or vector samples."""
    n, m = len(x), len(y)
    D = [[0.0] * m for _ in range(n)]
    for i in range(n):
        xi = x[i]
        for j in range(m):
            yj = y[j]
            if isinstance(xi, (list, tuple)):
                D[i][j] = sum((xi[k] - yj[k]) ** 2 for k in range(len(xi)))
            else:
                D[i][j] = (xi - yj) ** 2
    return D


def soft_dtw(x, y, gamma=1.0):
    """Soft-DTW discrepancy between series x and y with temperature gamma."""
    D = _cost_matrix(x, y)
    n, m = len(x), len(y)
    INF = float("inf")
    r = [[INF] * (m + 1) for _ in range(n + 1)]
    r[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r[i][j] = D[i - 1][j - 1] + _softmin(r[i - 1][j], r[i][j - 1], r[i - 1][j - 1], gamma)
    return r[n][m]


def _forward_full(x, y, gamma):
    """Return the full accumulated cost matrix r (with padding) and cost matrix D."""
    D = _cost_matrix(x, y)
    n, m = len(x), len(y)
    INF = float("inf")
    r = [[INF] * (m + 2) for _ in range(n + 2)]
    r[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r[i][j] = D[i - 1][j - 1] + _softmin(r[i - 1][j], r[i][j - 1], r[i - 1][j - 1], gamma)
    return r, D, n, m


def alignment_matrix(x, y, gamma=1.0):
    """Soft alignment matrix E[i][j] = expected occupancy of cell (i,j) over alignment paths.

    This is the gradient of soft-DTW with respect to the cost matrix (Cuturi-Blondel backward pass)."""
    r, D, n, m = _forward_full(x, y, gamma)
    INF = float("inf")
    # backward recursion (Cuturi-Blondel). Pad D by one extra row/col of zeros so the
    # successor lookups below are always in range; E is one bigger than r's interior.
    Dp = [[0.0] * (m + 2) for _ in range(n + 2)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            Dp[i][j] = D[i - 1][j - 1]
    E = [[0.0] * (m + 2) for _ in range(n + 2)]
    r[n + 1][m + 1] = r[n][m]
    Dp[n + 1][m + 1] = 0.0
    E[n + 1][m + 1] = 1.0
    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            a = math.exp((r[i + 1][j] - r[i][j] - Dp[i + 1][j]) / gamma) if r[i + 1][j] != INF else 0.0
            b = math.exp((r[i][j + 1] - r[i][j] - Dp[i][j + 1]) / gamma) if r[i][j + 1] != INF else 0.0
            c = math.exp((r[i + 1][j + 1] - r[i][j] - Dp[i + 1][j + 1]) / gamma) if r[i + 1][j + 1] != INF else 0.0
            E[i][j] = a * E[i + 1][j] + b * E[i][j + 1] + c * E[i + 1][j + 1]
    return [[E[i][j] for j in range(1, m + 1)] for i in range(1, n + 1)]


def hard_dtw(x, y):
    """Classic DTW with squared-Euclidean cost (independent reference; gamma -> 0 limit of soft-DTW)."""
    D = _cost_matrix(x, y)
    n, m = len(x), len(y)
    INF = float("inf")
    r = [[INF] * (m + 1) for _ in range(n + 1)]
    r[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r[i][j] = D[i - 1][j - 1] + min(r[i - 1][j], r[i][j - 1], r[i - 1][j - 1])
    return r[n][m]
