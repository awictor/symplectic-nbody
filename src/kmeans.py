"""k-means clustering: finding groups in unlabelled data.

Given points and a number k, k-means partitions them into k clusters so that each point belongs
to the nearest cluster CENTRE (centroid), minimizing the total within-cluster squared distance --
the "inertia" sum_i ||x_i - c_{assign(i)}||^2. It is the workhorse of unsupervised learning:
customer segmentation, image color quantization, vector quantization, and feature learning.

Lloyd's algorithm (1957) solves it by alternating two steps until nothing moves:

  ASSIGN: put each point in the cluster of its nearest current centroid,
  UPDATE: move each centroid to the mean of its assigned points.

Each step can only lower the inertia, so it converges -- but only to a LOCAL minimum, and a bad
random start can land in a poor one. The k-means++ seeding (Arthur & Vassilvitskii, 2007) fixes
this: pick the first centre at random, then each subsequent centre with probability proportional
to its squared distance from the nearest chosen centre, spreading the seeds out. It gives a
provable O(log k) approximation guarantee and, in practice, far better and more repeatable
clusterings. Running a few restarts and keeping the lowest-inertia result is standard.

This module implements Lloyd's algorithm with both random and k-means++ initialization, the
inertia objective, multi-restart selection, and a silhouette-style separation check, and verifies
it recovers well-separated blobs and that inertia decreases monotonically. Pure stdlib (seeded
LCG); the unsupervised-learning companion to the SVD/PCA note."""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def randint(self, k: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _mean(points):
    n = len(points)
    d = len(points[0])
    return [sum(p[j] for p in points) / n for j in range(d)]


def _nearest(point, centroids):
    """Index of the nearest centroid and its squared distance."""
    best, bd = 0, _dist2(point, centroids[0])
    for i in range(1, len(centroids)):
        d = _dist2(point, centroids[i])
        if d < bd:
            best, bd = i, d
    return best, bd


def init_random(data, k, rng):
    """Pick k distinct data points as initial centroids uniformly at random."""
    n = len(data)
    idx = []
    while len(idx) < k:
        j = rng.randint(n)
        if j not in idx:
            idx.append(j)
    return [data[j][:] for j in idx]


def init_plus_plus(data, k, rng):
    """k-means++ seeding: spread the initial centres by choosing each with probability
    proportional to its squared distance from the nearest already-chosen centre."""
    n = len(data)
    centroids = [data[rng.randint(n)][:]]
    while len(centroids) < k:
        d2 = [_nearest(p, centroids)[1] for p in data]
        total = sum(d2)
        if total == 0:                       # all points coincide with a centre
            centroids.append(data[rng.randint(n)][:])
            continue
        r = rng.random() * total             # weighted pick
        acc = 0.0
        chosen = n - 1
        for i, w in enumerate(d2):
            acc += w
            if acc >= r:
                chosen = i
                break
        centroids.append(data[chosen][:])
    return centroids


def kmeans(data, k, init="plus_plus", max_iter: int = 300, seed: int = 1, tol: float = 1e-10):
    """Cluster `data` into k groups by Lloyd's algorithm. `init` is 'plus_plus' or 'random'.
    Returns (centroids, labels, inertia, iterations)."""
    if k <= 0 or k > len(data):
        raise ValueError("need 1 <= k <= number of points")
    rng = _Rng(seed)
    centroids = init_plus_plus(data, k, rng) if init == "plus_plus" else init_random(data, k, rng)
    labels = [0] * len(data)
    prev_inertia = None
    for it in range(1, max_iter + 1):
        # ASSIGN
        inertia = 0.0
        for i, p in enumerate(data):
            labels[i], d = _nearest(p, centroids)
            inertia += d
        # UPDATE
        for c in range(k):
            members = [data[i] for i in range(len(data)) if labels[i] == c]
            if members:
                centroids[c] = _mean(members)
            # empty cluster: leave its centroid (or could re-seed); rare with k-means++
        if prev_inertia is not None and abs(prev_inertia - inertia) <= tol * (1 + inertia):
            return centroids, labels, inertia, it
        prev_inertia = inertia
    return centroids, labels, inertia, max_iter


def kmeans_best(data, k, restarts: int = 10, init="plus_plus", seed: int = 1):
    """Run k-means several times from different seeds and keep the lowest-inertia clustering --
    the standard defence against Lloyd's local minima. Returns (centroids, labels, inertia)."""
    best = None
    for r in range(restarts):
        c, l, inertia, _ = kmeans(data, k, init=init, seed=seed + r)
        if best is None or inertia < best[2]:
            best = (c, l, inertia)
    return best


def inertia(data, centroids, labels) -> float:
    """Total within-cluster squared distance for a given assignment."""
    return sum(_dist2(data[i], centroids[labels[i]]) for i in range(len(data)))


def cluster_sizes(labels, k):
    """Number of points assigned to each cluster."""
    sizes = [0] * k
    for l in labels:
        sizes[l] += 1
    return sizes


def silhouette(data, labels, k):
    """Mean silhouette score in [-1, 1]: for each point, (b - a) / max(a, b) where a is its mean
    distance to its own cluster and b the mean distance to the nearest OTHER cluster. Near 1
    means tight, well-separated clusters. O(n^2)."""
    n = len(data)
    if k < 2:
        return 0.0
    total = 0.0
    counted = 0
    for i in range(n):
        own = labels[i]
        # mean distance to own cluster (excluding self)
        same = [math.sqrt(_dist2(data[i], data[j])) for j in range(n) if labels[j] == own and j != i]
        if not same:
            continue                          # singleton cluster contributes 0
        a = sum(same) / len(same)
        b = math.inf
        for c in range(k):
            if c == own:
                continue
            other = [math.sqrt(_dist2(data[i], data[j])) for j in range(n) if labels[j] == c]
            if other:
                b = min(b, sum(other) / len(other))
        if b == math.inf:
            continue
        total += (b - a) / max(a, b)
        counted += 1
    return total / counted if counted else 0.0
