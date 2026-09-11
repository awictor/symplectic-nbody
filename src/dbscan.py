"""DBSCAN: density-based clustering that finds arbitrary shapes and flags noise.

k-means and Gaussian mixtures assume you know k and that clusters are roughly blobby. DBSCAN
(Ester, Kriegel, Sander, Xu, 1996) assumes neither: it finds clusters as connected regions of high
point density, discovers their number automatically, handles arbitrary shapes (two interlocking
moons, concentric rings), and explicitly labels outliers as NOISE rather than forcing every point
into a cluster.

Two parameters define "dense enough": a radius EPS and a count MIN_PTS. Each point is then one of

  CORE   -- has at least min_pts neighbours within eps (including itself),
  BORDER -- within eps of a core point but not itself core,
  NOISE  -- neither.

A cluster grows by starting from an unvisited core point and flood-filling: absorb its eps-
neighbours, and if any of them is also core, absorb THEIR neighbours too, transitively. Border
points join the first cluster that reaches them; noise points are left unlabeled (conventionally
-1). The whole thing needs no k and is driven entirely by local density, so an S-curve or a ring is
recovered as one cluster where k-means would slice it in half.

This module implements DBSCAN with Euclidean distance and the core/border/noise classification,
plus a k-distance helper for choosing eps -- verified that it separates two interlocking half-moons
that k-means cannot, isolates concentric rings, flags sparse outliers as noise, discovers the
cluster count on its own, and degenerates sensibly at extreme parameters. Pure stdlib; the
density-based companion to the k-means and Gaussian-mixture notes."""

from __future__ import annotations

import math

NOISE = -1


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def _region_query(X, i, eps2):
    """Indices of all points within eps of point i (including i itself)."""
    return [j for j in range(len(X)) if _dist2(X[i], X[j]) <= eps2]


def dbscan(X, eps, min_pts):
    """Cluster points by density.

    X: list of coordinate vectors. eps: neighbourhood radius. min_pts: density threshold
    (a point is core if it has >= min_pts neighbours within eps, counting itself).

    Returns a list of integer labels, one per point: 0,1,2,... for clusters, -1 (NOISE) for
    outliers. Cluster ids are assigned in discovery order."""
    n = len(X)
    eps2 = eps * eps
    labels = [None] * n            # None = unvisited
    cluster_id = -1

    for i in range(n):
        if labels[i] is not None:
            continue
        neighbours = _region_query(X, i, eps2)
        if len(neighbours) < min_pts:
            labels[i] = NOISE       # provisionally noise; may be claimed as a border point later
            continue
        # start a new cluster and flood-fill from this core point
        cluster_id += 1
        labels[i] = cluster_id
        seeds = [j for j in neighbours if j != i]
        k = 0
        while k < len(seeds):
            j = seeds[k]
            k += 1
            if labels[j] == NOISE:
                labels[j] = cluster_id           # border point of this cluster
            if labels[j] is not None:
                continue
            labels[j] = cluster_id
            j_neighbours = _region_query(X, j, eps2)
            if len(j_neighbours) >= min_pts:     # j is also core: extend the frontier
                seeds.extend(j_neighbours)

    return labels


def classify_points(X, eps, min_pts):
    """Label every point as 'core', 'border', or 'noise' (independent of cluster assignment)."""
    n = len(X)
    eps2 = eps * eps
    neigh = [_region_query(X, i, eps2) for i in range(n)]
    is_core = [len(neigh[i]) >= min_pts for i in range(n)]
    kinds = []
    for i in range(n):
        if is_core[i]:
            kinds.append("core")
        elif any(is_core[j] for j in neigh[i]):
            kinds.append("border")
        else:
            kinds.append("noise")
    return kinds


def n_clusters(labels):
    """Number of clusters found (excluding the noise label)."""
    return len({c for c in labels if c != NOISE})


def k_distances(X, k):
    """Sorted list of each point's distance to its k-th nearest neighbour.

    Plotting this sorted ascending gives the classic 'k-distance graph': the elbow is a good
    choice of eps (with min_pts = k+1)."""
    n = len(X)
    kd = []
    for i in range(n):
        dists = sorted(math.sqrt(_dist2(X[i], X[j])) for j in range(n) if j != i)
        if len(dists) >= k:
            kd.append(dists[k - 1])
    return sorted(kd)
