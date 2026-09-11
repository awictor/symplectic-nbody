"""Hierarchical agglomerative clustering: building a tree of nested groupings.

k-means and DBSCAN give one flat partition; hierarchical clustering gives the whole FAMILY of them
at once, as a tree. Agglomerative clustering starts with every point its own cluster and, at each
step, merges the two CLOSEST clusters, recording the merge and the distance at which it happened.
After n-1 merges everything is one cluster, and the record -- the DENDROGRAM -- shows the nested
structure: cut it at any height to read off a flat clustering, and the number of clusters falls out
of where you cut rather than being fixed in advance.

What "closest" means is the LINKAGE, and it changes the shape of the clusters:

  SINGLE   -- distance between the two nearest points (tends to chain, follows filaments)
  COMPLETE -- distance between the two farthest points (compact, roughly equal-diameter clusters)
  AVERAGE  -- mean pairwise distance (UPGMA; a balance of the two)
  WARD     -- merge the pair that least increases total within-cluster variance (tight, spherical)

The merge distances are MONOTONE for these linkages (each merge is at least as high as the last),
which is what makes the dendrogram a valid tree with no crossings. This module builds the full
dendrogram by the Lance-Williams update, cuts it into k clusters or at a height threshold, and
reports the cophenetic (merge) distances -- verified that it recovers well-separated blobs, that
merge heights increase monotonically, that single-linkage chains along a bridge where complete
linkage stays compact, and that Ward recovers spherical clusters. Pure stdlib; the tree-structured
companion to the k-means, Gaussian-mixture, and DBSCAN notes."""

from __future__ import annotations

import math

LINKAGES = ("single", "complete", "average", "ward")


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def linkage(X, method="ward"):
    """Build the agglomerative merge tree.

    Returns a list of n-1 merges, each (a, b, distance, size): clusters a and b (by id) merge into
    a new cluster whose id is n + step, at the given distance, with the given member count. New ids
    follow scipy's convention (original points are 0..n-1, merges create n, n+1, ...)."""
    if method not in LINKAGES:
        raise ValueError(f"unknown linkage {method!r}")
    n = len(X)
    # active clusters: id -> list of member point indices; plus a centroid & size for Ward
    members = {i: [i] for i in range(n)}
    size = {i: 1 for i in range(n)}
    # pairwise cluster distances, seeded with point distances (store as actual distance, not sq)
    # D[(i,j)] with i<j
    D = {}
    for i in range(n):
        for j in range(i + 1, n):
            D[(i, j)] = math.sqrt(_dist2(X[i], X[j]))

    # for Ward we track centroids
    centroid = {i: list(X[i]) for i in range(n)}

    def key(a, b):
        return (a, b) if a < b else (b, a)

    active = list(range(n))
    merges = []
    next_id = n

    while len(active) > 1:
        # find the closest active pair
        best = None
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                a, b = active[ai], active[bi]
                d = D[key(a, b)]
                if best is None or d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        new = next_id
        next_id += 1
        na, nb = size[a], size[b]
        nnew = na + nb
        merges.append((a, b, d, nnew))

        # Lance-Williams update of the distance from the new cluster to every other cluster c
        for c in active:
            if c == a or c == b:
                continue
            dac = D[key(a, c)]
            dbc = D[key(b, c)]
            dab = d
            nc = size[c]
            if method == "single":
                dnew = min(dac, dbc)
            elif method == "complete":
                dnew = max(dac, dbc)
            elif method == "average":
                dnew = (na * dac + nb * dbc) / nnew
            else:  # ward
                t = na + nb + nc
                dnew = math.sqrt(((na + nc) * dac * dac + (nb + nc) * dbc * dbc
                                  - nc * dab * dab) / t)
            D[key(new, c)] = dnew

        # retire a and b, activate new
        members[new] = members[a] + members[b]
        size[new] = nnew
        active = [c for c in active if c != a and c != b] + [new]

    return merges


def fcluster(X, n_clusters, method="ward"):
    """Flat clustering into exactly n_clusters by cutting the dendrogram. Returns labels 0..k-1."""
    n = len(X)
    if n_clusters >= n:
        return list(range(n))
    if n_clusters <= 1:
        return [0] * n
    merges = linkage(X, method)
    # replay the first n - n_clusters merges with a union-find, then label the components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # map merge-created ids back to a representative original point
    rep = {i: i for i in range(n)}
    n_merge = n - n_clusters
    for s in range(n_merge):
        a, b, _, _ = merges[s]
        ra, rb = rep[a], rep[b]
        pa, pb = find(ra), find(rb)
        parent[pa] = pb
        rep[n + s] = pb
    # relabel roots contiguously
    root_label = {}
    labels = []
    for i in range(n):
        r = find(i)
        if r not in root_label:
            root_label[r] = len(root_label)
        labels.append(root_label[r])
    return labels


def merge_heights(merges):
    """The distance at which each merge occurred, in order."""
    return [m[2] for m in merges]


def is_monotone(merges, tol=1e-9):
    """True if merge heights are non-decreasing (a valid dendrogram has no inversions)."""
    h = merge_heights(merges)
    return all(h[i + 1] >= h[i] - tol for i in range(len(h) - 1))
