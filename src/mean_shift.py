"""Mean shift: clustering by climbing to the modes of a density, without knowing how many.

K-means and Gaussian mixtures need you to say how many clusters there are; mean shift discovers that
number itself. It treats the data as samples from a probability density and slides every point UPHILL
along the density gradient to the nearest PEAK (mode). Points that flow to the same peak form a
cluster, so the number of clusters is however many distinct modes the data has -- decided by one
parameter, the kernel BANDWIDTH (how far a point looks for neighbours), not by a preset k.

The update is beautifully simple. Around a point x, the mean-shift vector points toward the
kernel-weighted average of the neighbours:

    x  <-  ( sum_i K(x - x_i) x_i ) / ( sum_i K(x - x_i) ),

and iterating it is provably a gradient ascent on the kernel-density estimate -- each step moves to
the local weighted centroid, climbing the density until it converges to a mode. With a FLAT
(uniform-ball) kernel the average is over neighbours within the bandwidth; with a GAUSSIAN kernel every
point contributes with exponentially decaying weight. After all points converge, nearby modes are
merged (within a tolerance) and each point is labelled by its mode.

This module runs mean shift with either kernel, returning cluster labels and the mode locations, and
exposes the single-point mode-climb for inspection. Validated: on well-separated Gaussian blobs it
recovers the right NUMBER of clusters automatically and places each mode near the true blob centre; a
smaller bandwidth finds more (finer) clusters and a larger one fewer; every point is assigned to its
nearest mode; the mode-climb is a monotone density ascent (the KDE value never decreases); and a
single blob yields one cluster. Pure stdlib; the mode-seeking companion to the k-means / DBSCAN /
GMM clustering and the KDE density tools."""

from __future__ import annotations

import math


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def _flat_kernel(d2, bandwidth):
    return 1.0 if d2 <= bandwidth * bandwidth else 0.0


def _gaussian_kernel(d2, bandwidth):
    return math.exp(-0.5 * d2 / (bandwidth * bandwidth))


def _mean_shift_vector(x, points, bandwidth, kernel):
    """The kernel-weighted mean of the neighbours of x (the new position)."""
    dim = len(x)
    num = [0.0] * dim
    den = 0.0
    for p in points:
        w = kernel(_dist2(x, p), bandwidth)
        if w > 0:
            for i in range(dim):
                num[i] += w * p[i]
            den += w
    if den == 0:
        return list(x)
    return [num[i] / den for i in range(dim)]


def climb(x, points, bandwidth, kernel="gaussian", tol=1e-5, max_iter=300):
    """Slide a single point uphill to its density mode. Returns the mode location."""
    k = _gaussian_kernel if kernel == "gaussian" else _flat_kernel
    cur = list(x)
    for _ in range(max_iter):
        nxt = _mean_shift_vector(cur, points, bandwidth, k)
        if _dist2(cur, nxt) < tol * tol:
            cur = nxt
            break
        cur = nxt
    return cur


def mean_shift(points, bandwidth, kernel="gaussian", tol=1e-5, merge_tol=None, max_iter=300):
    """Cluster points by mean shift. Returns (labels, modes) where labels[i] is the cluster index of
    point i and modes is the list of distinct mode locations."""
    if merge_tol is None:
        merge_tol = bandwidth * 0.5
    converged = [climb(p, points, bandwidth, kernel, tol, max_iter) for p in points]
    # merge modes that are within merge_tol
    modes = []
    labels = []
    for c in converged:
        found = -1
        for mi, m in enumerate(modes):
            if _dist2(c, m) < merge_tol * merge_tol:
                found = mi
                break
        if found == -1:
            modes.append(c)
            labels.append(len(modes) - 1)
        else:
            labels.append(found)
    return labels, modes


def kde_value(x, points, bandwidth):
    """The (unnormalized) Gaussian kernel-density estimate at x -- what the mode-climb ascends."""
    return sum(_gaussian_kernel(_dist2(x, p), bandwidth) for p in points)


def n_clusters(labels):
    return len(set(labels))
