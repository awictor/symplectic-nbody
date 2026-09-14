"""Isolation Forest: unsupervised anomaly detection by how few random cuts it takes to isolate a point.

Most anomaly detectors model what NORMAL looks like and flag deviations, which is expensive in high
dimensions. Isolation Forest (Liu, Ting, Zhou 2008) inverts the idea: anomalies are FEW and DIFFERENT,
so they are easy to isolate. Build a random binary tree by repeatedly picking a random feature and a
random split value between that feature's min and max, partitioning the points, and recursing. Points
in dense regions need many cuts to be separated from their neighbours; an outlier sitting alone in
empty space gets cut off after just a few. So the PATH LENGTH from the root to a point -- the number of
splits to isolate it -- is short for anomalies and long for normal points.

Average the path length over a forest of such random trees (each grown on a small random subsample,
the classic choice being 256 points) and normalize by the expected path length of an unsuccessful
binary-search-tree lookup, c(n) = 2 H(n-1) - 2(n-1)/n where H is the harmonic number. The anomaly
score is s = 2^(-E[path]/c(n)): scores near 1 mean "isolated quickly, likely anomaly", scores well
below 0.5 mean "buried in a cluster, normal". No distance metric, no density estimate, and it scales
linearly -- which is why it is a workhorse for fraud and intrusion detection.

This module builds an isolation forest with a seeded LCG (so runs are reproducible), scores points by
their normalized average path length, and exposes the raw path lengths. It is validated: planted
outliers receive strictly higher anomaly scores than the inlier cluster; the score of a point far from
every training point exceeds that of a central one; scores lie in (0, 1); the c(n) normalization
matches its closed form; a single well-separated outlier is ranked the most anomalous; results are
reproducible for a fixed seed and vary with it; and it works in 1-D through higher dimensions. Pure
stdlib; the unsupervised-anomaly companion to the LOF, k-means, and matrix-profile tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u32(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def uniform(self):
        return (self.next_u32() >> 8) / (1 << 24)

    def randint(self, lo, hi):
        # inclusive lo, exclusive hi
        return lo + int(self.uniform() * (hi - lo)) % (hi - lo)


class _Node:
    __slots__ = ("feature", "split", "left", "right", "size", "depth")

    def __init__(self):
        self.feature = None
        self.split = None
        self.left = None
        self.right = None
        self.size = 0
        self.depth = 0


def _c(n):
    """Expected path length of an unsuccessful search in a binary search tree of n points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # harmonic number H(n-1): exact sum for small n, Euler approximation for large n
    if n <= 1000:
        H = sum(1.0 / k for k in range(1, n))
    else:
        H = math.log(n - 1) + 0.5772156649015329 + 1.0 / (2 * (n - 1))
    return 2.0 * H - 2.0 * (n - 1) / n


def _build_tree(points, rng, depth, max_depth):
    node = _Node()
    node.size = len(points)
    node.depth = depth
    if depth >= max_depth or len(points) <= 1:
        return node
    dim = len(points[0])
    # pick a random feature that actually varies
    feats = list(range(dim))
    # shuffle-lite: try features in a random rotation to find one with spread
    start = rng.randint(0, dim)
    chosen = None
    for off in range(dim):
        f = (start + off) % dim
        lo = min(p[f] for p in points)
        hi = max(p[f] for p in points)
        if hi > lo:
            chosen = (f, lo, hi)
            break
    if chosen is None:
        return node                                  # all identical points: a leaf
    f, lo, hi = chosen
    split = lo + rng.uniform() * (hi - lo)
    left = [p for p in points if p[f] < split]
    right = [p for p in points if p[f] >= split]
    if not left or not right:
        return node
    node.feature = f
    node.split = split
    node.left = _build_tree(left, rng, depth + 1, max_depth)
    node.right = _build_tree(right, rng, depth + 1, max_depth)
    return node


def _path_length(node, point):
    """Path length to isolate `point`, adding the BST correction c(size) at an early-terminated leaf."""
    length = 0
    while node.feature is not None:
        if point[node.feature] < node.split:
            node = node.left
        else:
            node = node.right
        length += 1
    return length + _c(node.size)


class IsolationForest:
    """A forest of random isolation trees for unsupervised anomaly scoring."""

    def __init__(self, n_trees=100, sample_size=256, seed=0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self.trees = []
        self.n_train = 0

    def fit(self, X):
        X = [list(p) for p in X]
        n = len(X)
        self.n_train = n
        psi = min(self.sample_size, n)
        self.c_psi = _c(psi)
        max_depth = max(1, int(math.ceil(math.log2(psi)))) if psi > 1 else 1
        rng = _Rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            # subsample psi points (with a simple random pick, without replacement)
            if psi < n:
                idx = set()
                while len(idx) < psi:
                    idx.add(rng.randint(0, n))
                sample = [X[i] for i in idx]
            else:
                sample = list(X)
            self.trees.append(_build_tree(sample, rng, 0, max_depth))
        return self

    def path_length(self, point):
        """Mean path length of `point` across the forest."""
        return sum(_path_length(t, point) for t in self.trees) / len(self.trees)

    def anomaly_score(self, point):
        """Anomaly score s = 2^(-E[path]/c(psi)) in (0, 1); higher = more anomalous."""
        e = self.path_length(list(point))
        return 2.0 ** (-e / self.c_psi) if self.c_psi > 0 else 0.5

    def score_samples(self, X):
        return [self.anomaly_score(p) for p in X]

    def rank(self, X):
        """Indices of X sorted from most to least anomalous."""
        scores = self.score_samples(X)
        return sorted(range(len(X)), key=lambda i: -scores[i])
