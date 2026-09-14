"""Fuzzy c-means: soft clustering where every point belongs to every cluster, by graded membership.

Hard clustering (k-means) forces each point into exactly one cluster -- a crisp choice that throws away
information about points sitting BETWEEN clusters, on a boundary, or in an overlap. FUZZY C-MEANS (Dunn
1973, Bezdek 1981) instead gives every point a MEMBERSHIP DEGREE in each cluster, a set of weights that
sum to 1: a point deep inside a cluster is ~100% that cluster, while a point halfway between two centers
might be 50/50. It minimizes the fuzzy objective

    J = sum_i sum_j  u_{ij}^m  ||x_i - c_j||^2,       sum_j u_{ij} = 1,

where m > 1 is the FUZZIFIER controlling softness (m -> 1 recovers hard k-means; larger m blurs the
memberships toward uniform). Alternating optimization solves it: with centers fixed, the optimal
memberships have the closed form u_{ij} = 1 / sum_k (d_{ij}/d_{ik})^{2/(m-1)}; with memberships fixed, each
center is the membership-weighted mean of all points, c_j = sum_i u_{ij}^m x_i / sum_i u_{ij}^m. Iterating
the two steps monotonically decreases J to a local minimum.

Fuzzy c-means is used wherever cluster boundaries are genuinely gradual -- image segmentation, medical
diagnosis, geology, and as a soft front end to classifiers. The membership vector also flags ambiguous
points (near-uniform memberships) and outliers.

This module runs fuzzy c-means with a seeded initialization, returns the cluster centers and the full
membership matrix, computes the objective and the fuzzy partition coefficient (a cluster-validity score),
and hardens memberships to labels. It is validated: on well-separated blobs it recovers the centers and
assigns near-crisp memberships (~1 for the right cluster); a point exactly between two centers gets ~50/50
membership; memberships always sum to 1; the objective J decreases monotonically each iteration; as the
fuzzifier m -> 1 the result approaches hard k-means (cross-checked against the repo's k-means); the
partition coefficient is near 1 for crisp data and lower for fuzzy data; and results are reproducible per
seed. Pure stdlib; the soft-clustering companion to the k-means, GMM, affinity-propagation, and DBSCAN
tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def _init_memberships(n, c, rng):
    """Random membership matrix (n x c), each row normalized to sum 1."""
    U = []
    for _ in range(n):
        row = [rng.u() + 1e-6 for _ in range(c)]
        s = sum(row)
        U.append([v / s for v in row])
    return U


def _update_centers(data, U, m):
    """Membership-weighted cluster centers. c_j = sum_i u_ij^m x_i / sum_i u_ij^m."""
    n = len(data)
    c = len(U[0])
    dim = len(data[0])
    centers = []
    for j in range(c):
        num = [0.0] * dim
        den = 0.0
        for i in range(n):
            w = U[i][j] ** m
            den += w
            xi = data[i]
            for d in range(dim):
                num[d] += w * xi[d]
        centers.append([num[d] / den if den > 0 else 0.0 for d in range(dim)])
    return centers


def _update_memberships(data, centers, m):
    """Closed-form optimal memberships given centers. Handles points coincident with a center."""
    n = len(data)
    c = len(centers)
    power = 2.0 / (m - 1.0)
    U = []
    for i in range(n):
        dists = [_dist2(data[i], centers[j]) for j in range(c)]
        # if a point coincides with one or more centers, split membership among them
        zeros = [j for j in range(c) if dists[j] == 0.0]
        if zeros:
            row = [0.0] * c
            for j in zeros:
                row[j] = 1.0 / len(zeros)
            U.append(row)
            continue
        # u_ij = 1 / sum_k (d_ij/d_ik)^{1/(m-1)}; dists are SQUARED, so the exponent is power/2
        # where power = 2/(m-1).
        exp = power / 2.0
        row = []
        for j in range(c):
            s = sum((dists[j] / dists[k]) ** exp for k in range(c))
            row.append(1.0 / s)
        U.append(row)
    return U


def objective(data, centers, U, m):
    """Fuzzy objective J = sum_i sum_j u_ij^m ||x_i - c_j||^2."""
    n = len(data)
    c = len(centers)
    total = 0.0
    for i in range(n):
        for j in range(c):
            total += (U[i][j] ** m) * _dist2(data[i], centers[j])
    return total


def fuzzy_cmeans(data, c, m=2.0, max_iter=300, tol=1e-6, seed=1, track=False):
    """Fuzzy c-means clustering. Returns a dict with centers, memberships (n x c), labels
    (hardened argmax), objective, iterations, and partition_coefficient.

    m > 1 is the fuzzifier (2.0 standard). Alternates center and membership updates until J converges."""
    if m <= 1.0:
        raise ValueError("fuzzifier m must be > 1")
    n = len(data)
    rng = _Rng(seed)
    U = _init_memberships(n, c, rng)
    centers = _update_centers(data, U, m)
    prev_j = None
    history = []
    iters = max_iter
    for it in range(max_iter):
        centers = _update_centers(data, U, m)
        U = _update_memberships(data, centers, m)
        j = objective(data, centers, U, m)
        history.append(j)
        if prev_j is not None and abs(prev_j - j) < tol * (1 + abs(prev_j)):
            iters = it + 1
            break
        prev_j = j

    labels = [max(range(c), key=lambda k: U[i][k]) for i in range(n)]
    pc = partition_coefficient(U)
    res = {
        "centers": centers,
        "memberships": U,
        "labels": labels,
        "objective": objective(data, centers, U, m),
        "iterations": iters,
        "partition_coefficient": pc,
    }
    if track:
        res["history"] = history
    return res


def partition_coefficient(U):
    """Fuzzy partition coefficient: mean over points of sum_j u_ij^2. Near 1 for crisp partitions,
    down to 1/c for maximally fuzzy ones. A simple cluster-validity index."""
    n = len(U)
    c = len(U[0])
    total = 0.0
    for i in range(n):
        total += sum(U[i][j] ** 2 for j in range(c))
    return total / n


def predict_membership(x, centers, m=2.0):
    """Membership vector of a new point x given fitted centers."""
    return _update_memberships([x], centers, m)[0]
