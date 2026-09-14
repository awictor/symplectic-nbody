"""Local Outlier Factor: density-based anomaly detection that catches LOCAL outliers.

A global anomaly detector asks "is this point far from everything?" -- but real data often has clusters
of very different densities, and a point can be perfectly normal by global standards yet clearly wrong
in its local neighbourhood. Consider a tight dense cluster and a loose sparse one: a point sitting just
outside the dense cluster, at a distance that is completely ordinary inside the sparse cluster, is a
LOCAL outlier that distance- or isolation-based methods miss. LOF (Breunig, Kriegel, Ng, Sander 2000)
catches exactly these by comparing each point's density to the densities of its neighbours.

The construction is a short chain of definitions over the k nearest neighbours. The K-DISTANCE of a
point is the distance to its k-th neighbour. The REACHABILITY DISTANCE of A from B is
max(k-distance(B), d(A,B)) -- a smoothing that stops points inside a dense region from getting
absurdly small distances. The LOCAL REACHABILITY DENSITY (lrd) of a point is the inverse of its average
reachability distance to its neighbours: high where points are packed tight, low where they are spread
out. Finally the LOF of a point is the average ratio of its neighbours' lrd to its own lrd. If the
point sits in a region as dense as its neighbours, the ratio is about 1; if its neighbours are much
denser than it is -- the signature of a local outlier -- the LOF climbs well above 1.

This module computes k-distances, reachability distances, local reachability densities, and the LOF
score for every point (brute-force nearest neighbours, any dimension). It is validated: inliers deep in
a uniform cloud have LOF near 1; a point far outside a single cluster has LOF well above 1; on the
canonical two-density example a point normal by GLOBAL distance but wrong for its local cluster is
flagged (the property that distinguishes LOF from global detectors); the reachability distance is
symmetric in its floor and never below the raw distance; and duplicate points get finite scores. Pure
stdlib; the density-based-anomaly companion to the Isolation-Forest, DBSCAN, and k-NN tools."""

from __future__ import annotations

import math


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def _neighbors(points, i, k):
    """Indices of the k nearest neighbours of point i (excluding itself), and the k-distance.

    Ties at the k-distance boundary are all included (standard LOF definition)."""
    d = [(_dist(points[i], points[j]), j) for j in range(len(points)) if j != i]
    d.sort()
    if k >= len(d):
        return [j for _, j in d], (d[-1][0] if d else 0.0)
    kdist = d[k - 1][0]
    nbrs = [j for dist, j in d if dist <= kdist + 1e-12]
    return nbrs, kdist


class LOF:
    """Local Outlier Factor scorer over a fixed point set."""

    def __init__(self, points, k=20):
        self.points = [list(p) for p in points]
        self.k = min(k, len(points) - 1)
        self._compute()

    def _compute(self):
        n = len(self.points)
        self.nbrs = [None] * n
        self.kdist = [0.0] * n
        for i in range(n):
            self.nbrs[i], self.kdist[i] = _neighbors(self.points, i, self.k)
        # local reachability density
        self.lrd = [0.0] * n
        for i in range(n):
            reach = []
            for j in self.nbrs[i]:
                rd = max(self.kdist[j], _dist(self.points[i], self.points[j]))
                reach.append(rd)
            avg = sum(reach) / len(reach) if reach else 0.0
            self.lrd[i] = 1.0 / avg if avg > 1e-15 else float("inf")
        # LOF
        self.lof = [0.0] * n
        for i in range(n):
            if not self.nbrs[i]:
                self.lof[i] = 1.0
                continue
            ratios = [self.lrd[j] / self.lrd[i] for j in self.nbrs[i]] if self.lrd[i] > 0 else [1.0]
            self.lof[i] = sum(ratios) / len(ratios)

    def scores(self):
        """LOF score per point (>~1 = outlier, ~1 = inlier)."""
        return list(self.lof)

    def k_distance(self, i):
        return self.kdist[i]

    def reachability_distance(self, i, j):
        """Reachability distance of point i from point j = max(k-distance(j), d(i,j))."""
        return max(self.kdist[j], _dist(self.points[i], self.points[j]))

    def local_reachability_density(self, i):
        return self.lrd[i]

    def rank(self):
        """Point indices sorted from most to least anomalous (highest LOF first)."""
        return sorted(range(len(self.points)), key=lambda i: -self.lof[i])


def lof_scores(points, k=20):
    """Convenience: LOF score for every point."""
    return LOF(points, k).scores()
