"""Matrix profile: the distance to every subsequence's nearest neighbour, and the motifs and discords it reveals.

Given a long time series and a window length m, slide the window over the series to get all
length-m SUBSEQUENCES. The MATRIX PROFILE stores, for each subsequence, the z-normalized Euclidean
distance to its nearest neighbour elsewhere in the series, along with the index of that neighbour (the
profile index). This single vector is astonishingly informative: its LOWEST values mark MOTIFS -- the
pair of subsequences most similar to each other, i.e. a repeated pattern -- and its HIGHEST values mark
DISCORDS, the most unusual subsequence, the best parameter-free anomaly detector known for time series.

The distances are Z-NORMALIZED (each window is standardized to zero mean and unit variance before
comparing), so the matrix profile matches shapes regardless of offset or scale -- a slow drift and a
sharp spike of the same shape are recognized as the same pattern. Computed naively this is O(n^2 m):
n subsequences each compared to n others at cost m. The MASS trick collapses the inner cost: a
z-normalized distance can be written in terms of the sliding DOT PRODUCT of the query against the
series, and all n sliding dot products are one convolution, done by FFT in O(n log n). So one
query-to-all distance profile costs O(n log n) instead of O(n m), and the STAMP algorithm builds the
full matrix profile by evaluating one such profile per subsequence -- with a mandatory EXCLUSION ZONE
around the diagonal so a window is never matched to itself or its trivial overlapping neighbours.

This module computes the matrix profile by MASS (FFT-accelerated) and by a transparent brute-force
reference, and extracts the top motif pair and the top discord. It is validated: the MASS matrix
profile matches the brute-force one to machine precision; a planted repeated pattern is recovered as
the motif; a planted anomaly is recovered as the discord; z-normalization makes the profile invariant
to adding a constant or scaling the series; and a single sliding distance profile matches direct
z-normalized Euclidean distances. Reuses the repo's FFT. Pure stdlib; the time-series companion to the
DTW, FFT, and cross-correlation tools."""

from __future__ import annotations

import math

from fft import convolve


def _sliding_stats(t, m):
    """Rolling mean and standard deviation of every length-m window of t. O(n)."""
    n = len(t)
    k = n - m + 1
    csum = [0.0] * (n + 1)
    csum2 = [0.0] * (n + 1)
    for i in range(n):
        csum[i + 1] = csum[i] + t[i]
        csum2[i + 1] = csum2[i] + t[i] * t[i]
    mean = [0.0] * k
    std = [0.0] * k
    for i in range(k):
        s = csum[i + m] - csum[i]
        s2 = csum2[i + m] - csum2[i]
        mu = s / m
        var = s2 / m - mu * mu
        mean[i] = mu
        std[i] = math.sqrt(var) if var > 1e-14 else 0.0
    return mean, std


def _sliding_dot(t, query):
    """Sliding dot product of `query` (length m) against every length-m window of t, via FFT.

    Returns a list of length n-m+1 where entry i = sum_j t[i+j] * query[j]."""
    m = len(query)
    n = len(t)
    # convolution of t with reversed query gives correlation; pick the valid full-overlap lags
    conv = convolve(list(t), list(reversed(query)))
    # conv[k] = sum_j t[j] * query_rev[k-j]; the window starting at i corresponds to k = i + m - 1
    return [conv[i + m - 1] for i in range(n - m + 1)]


def mass(t, query):
    """MASS: z-normalized Euclidean distance from `query` to every length-m subsequence of t.

    Returns a distance profile of length n-m+1. Uses the FFT sliding dot product."""
    m = len(query)
    n = len(t)
    k = n - m + 1
    qmu = sum(query) / m
    qvar = sum(x * x for x in query) / m - qmu * qmu
    qstd = math.sqrt(qvar) if qvar > 1e-14 else 0.0
    mean, std = _sliding_stats(t, m)
    dots = _sliding_dot(t, query)
    dist = [0.0] * k
    for i in range(k):
        if std[i] == 0.0 or qstd == 0.0:
            # a constant window: distance 0 iff both constant, else the max
            dist[i] = 0.0 if (std[i] == 0.0 and qstd == 0.0) else math.sqrt(2 * m)
            continue
        # z-normalized squared distance = 2m(1 - (dots - m*mean*qmu)/(m*std*qstd))
        corr = (dots[i] - m * mean[i] * qmu) / (m * std[i] * qstd)
        corr = max(-1.0, min(1.0, corr))
        d2 = 2 * m * (1 - corr)
        dist[i] = math.sqrt(max(d2, 0.0))
    return dist


def _znorm(w):
    m = len(w)
    mu = sum(w) / m
    var = sum(x * x for x in w) / m - mu * mu
    sd = math.sqrt(var) if var > 1e-14 else 0.0
    if sd == 0.0:
        return [0.0] * m
    return [(x - mu) / sd for x in w]


def brute_distance_profile(t, query):
    """Reference: z-normalized Euclidean distance from query to each window, computed directly."""
    m = len(query)
    n = len(t)
    qz = _znorm(query)
    out = []
    for i in range(n - m + 1):
        wz = _znorm(t[i:i + m])
        out.append(math.sqrt(sum((wz[j] - qz[j]) ** 2 for j in range(m))))
    return out


def matrix_profile(t, m, use_mass=True):
    """Full self-join matrix profile of series t with window length m.

    Returns (profile, index) where profile[i] is the z-normalized distance from window i to its
    nearest non-trivial neighbour and index[i] is that neighbour's start position."""
    n = len(t)
    k = n - m + 1
    excl = max(1, m // 2)                          # exclusion zone radius (trivial-match guard)
    profile = [math.inf] * k
    index = [-1] * k
    for i in range(k):
        query = t[i:i + m]
        dp = mass(t, query) if use_mass else brute_distance_profile(t, query)
        for j in range(k):
            if abs(i - j) < excl:
                continue
            if dp[j] < profile[i]:
                profile[i] = dp[j]
                index[i] = j
    return profile, index


def top_motif(t, m):
    """The motif: the pair of windows with the smallest matrix-profile distance. Returns (i, j, dist)."""
    profile, index = matrix_profile(t, m)
    best_i = min(range(len(profile)), key=lambda i: profile[i])
    return best_i, index[best_i], profile[best_i]


def top_discord(t, m):
    """The discord (anomaly): the window with the largest matrix-profile distance. Returns (i, dist)."""
    profile, index = matrix_profile(t, m)
    finite = [i for i in range(len(profile)) if profile[i] != math.inf]
    best_i = max(finite, key=lambda i: profile[i])
    return best_i, profile[best_i]
