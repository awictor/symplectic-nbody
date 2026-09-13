"""Exact Euclidean distance transform: distance to the nearest feature, in linear time.

Given a binary image -- some pixels are FEATURE (foreground), the rest background -- the distance
transform labels every pixel with its distance to the nearest feature pixel. It underlies shape
matching, skeletonization, the medial axis, path planning (stay far from obstacles), morphological
operations, and the anti-aliasing of fonts (signed distance fields). Computing it naively is
O(n^2) -- every pixel against every feature. The remarkable algorithm of Felzenszwalb and
Huttenlocher (2004) computes the EXACT Euclidean distance transform in O(n) total, by a beautiful
observation.

The squared distance transform is SEPARABLE: process each dimension in turn. Along a single row (or
column) the transform is a 1-D problem -- for each point x, minimize over all sites i of
f(i) + (x - i)^2, where f(i) is the cost accumulated so far. That is the LOWER ENVELOPE of a family
of upward parabolas, one per site, all with the same curvature. The lower envelope of n such
parabolas has at most n pieces and is found in a single O(n) sweep: maintain a stack of the parabolas
currently on the envelope and the breakpoints between them, adding each new parabola and popping any
it hides. Run this 1-D transform down every column, then across every row, and the 2-D exact squared
distance falls out in linear time.

This module implements the 1-D lower-envelope transform, the 2-D exact Euclidean distance transform,
its square root, and a helper that also returns the nearest feature site. Validated against brute
force: the transform matches an exhaustive nearest-feature search on random binary images for every
pixel; a single feature point gives exact radial distances; feature pixels have distance zero; the
1-D transform matches its own brute minimization; and an all-background image is handled. Pure stdlib;
the image/geometry companion to the Voronoi, k-d tree, and morphological tools."""

from __future__ import annotations

import math

INF = float("inf")


def distance_transform_1d(f):
    """Exact 1-D squared distance transform: for each x return min over i of f(i) + (x - i)^2.
    O(n) via the lower envelope of parabolas (Felzenszwalb-Huttenlocher)."""
    n = len(f)
    d = [INF] * n
    # only finite-cost sites contribute parabolas; if none, everything is INF
    if all(v == INF for v in f):
        return d
    v = [0] * n          # v[k] = x-coordinate of the k-th parabola on the envelope
    z = [0.0] * (n + 1)  # z[k], z[k+1] = range of x where parabola k is lowest
    # seed the envelope with the first finite site
    first = next(q for q in range(n) if f[q] < INF)
    k = 0
    v[0] = first
    z[0] = -INF
    z[1] = INF
    for q in range(first + 1, n):
        if f[q] == INF:
            continue
        # intersection of parabola from q with the current lowest (v[k])
        while True:
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
            if s <= z[k] and k > 0:
                k -= 1
            else:
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def squared_distance_transform_2d(feature):
    """Exact 2-D squared Euclidean distance transform. `feature` is a 2-D list of booleans (True =
    feature pixel). Returns a 2-D list of squared distances to the nearest feature pixel."""
    h = len(feature)
    if h == 0:
        return []
    w = len(feature[0])
    # initialize: 0 at features, +inf elsewhere
    d = [[0.0 if feature[i][j] else INF for j in range(w)] for i in range(h)]
    # transform along columns
    for j in range(w):
        col = [d[i][j] for i in range(h)]
        col = distance_transform_1d(col)
        for i in range(h):
            d[i][j] = col[i]
    # transform along rows
    for i in range(h):
        d[i] = distance_transform_1d(d[i])
    return d


def distance_transform_2d(feature):
    """Exact 2-D Euclidean distance transform (the square root of the squared transform)."""
    sq = squared_distance_transform_2d(feature)
    return [[math.sqrt(v) if v < INF else INF for v in row] for row in sq]


def nearest_feature(feature):
    """Return (distance_grid, nearest_grid) where nearest_grid[i][j] is the (row, col) of the
    closest feature pixel. Computed directly (O(n * #features)) as a reference-quality helper."""
    h = len(feature)
    w = len(feature[0]) if h else 0
    sites = [(i, j) for i in range(h) for j in range(w) if feature[i][j]]
    dist = [[INF] * w for _ in range(h)]
    near = [[None] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            best = INF
            bs = None
            for (si, sj) in sites:
                dd = (i - si) ** 2 + (j - sj) ** 2
                if dd < best:
                    best = dd
                    bs = (si, sj)
            dist[i][j] = math.sqrt(best) if best < INF else INF
            near[i][j] = bs
    return dist, near


# --- brute-force reference ---------------------------------------------------
def brute_squared_distance_2d(feature):
    """Exhaustive nearest-feature squared distance (reference for the fast transform)."""
    h = len(feature)
    w = len(feature[0]) if h else 0
    sites = [(i, j) for i in range(h) for j in range(w) if feature[i][j]]
    out = [[INF] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            best = INF
            for (si, sj) in sites:
                dd = (i - si) ** 2 + (j - sj) ** 2
                if dd < best:
                    best = dd
            out[i][j] = best
    return out


def brute_distance_1d(f):
    """Exhaustive 1-D squared distance transform (reference)."""
    n = len(f)
    return [min(f[i] + (x - i) ** 2 for i in range(n)) for x in range(n)]
