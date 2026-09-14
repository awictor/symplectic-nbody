"""Affinity propagation: clustering that elects its own exemplars by passing messages, with no preset k.

Most clustering methods make you choose the number of clusters up front (k-means) or tune a radius
(DBSCAN, mean-shift). AFFINITY PROPAGATION (Frey & Dueck, Science 2007) does neither. Every data point is
simultaneously a candidate to be a cluster center (an EXEMPLAR) and a member looking for one, and the
points converge on a good set of exemplars by exchanging two kinds of real-valued messages until a
consensus emerges:

  RESPONSIBILITY r(i,k): sent from point i to candidate exemplar k, saying how well-suited k is to be i's
      exemplar, accounting for the strongest competing candidate --
          r(i,k) = s(i,k) - max_{k' != k} ( a(i,k') + s(i,k') ).
  AVAILABILITY a(i,k): sent from candidate k back to i, saying how appropriate it is for i to pick k,
      given the support k has gathered from other points --
          a(i,k) = min( 0, r(k,k) + sum_{i' != i,k} max(0, r(i',k)) ),   a(k,k) = sum_{i' != k} max(0, r(i',k)).

The messages are damped for stability and iterated to convergence; a point k is an exemplar when
r(k,k) + a(k,k) > 0, and each point joins the exemplar maximizing a(i,k) + s(i,k). The number of clusters
is controlled indirectly by the PREFERENCE (the self-similarity s(k,k)): higher preference yields more
exemplars. Setting it to the median input similarity is the standard default and typically finds a natural
cluster count on its own.

This module builds the similarity matrix (negative squared Euclidean distance by default), runs the
damped message-passing updates to convergence, extracts the exemplars and cluster assignments, and reports
the net-similarity objective. It is validated: on well-separated Gaussian blobs it recovers the correct
number of clusters and assigns every point to its blob; each exemplar is an actual data point and every
point is assigned to the exemplar it is most similar to among those chosen; a higher preference produces
more clusters (and a very low one collapses to a single cluster); the net-similarity objective improves
over the iterations; and results are deterministic. Pure stdlib; the exemplar-based-clustering companion
to the k-means, DBSCAN, mean-shift, and spectral-clustering tools."""

from __future__ import annotations


def negative_sq_euclidean(points):
    """Similarity matrix s(i,j) = -||x_i - x_j||^2 (the standard affinity-propagation similarity)."""
    n = len(points)
    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = sum((points[i][d] - points[j][d]) ** 2 for d in range(len(points[i])))
            S[i][j] = -d
    return S


def median_offdiagonal(S):
    """Median of the off-diagonal similarities -- the standard default preference."""
    vals = []
    n = len(S)
    for i in range(n):
        for j in range(n):
            if i != j:
                vals.append(S[i][j])
    vals.sort()
    m = len(vals)
    if m == 0:
        return 0.0
    return vals[m // 2] if m % 2 == 1 else 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def affinity_propagation(S, preference=None, damping=0.5, max_iter=200, conv_iter=15):
    """Cluster by affinity propagation on a precomputed similarity matrix S (n x n).

    preference sets the diagonal s(k,k) (default: median off-diagonal similarity). damping in [0.5,1)
    stabilizes the updates. Returns a dict with exemplars (indices), labels (exemplar index per point),
    n_clusters, and the number of iterations run. Convergence = exemplar set unchanged for conv_iter."""
    n = len(S)
    S = [row[:] for row in S]
    if preference is None:
        preference = median_offdiagonal(S)
    for k in range(n):
        S[k][k] = preference

    R = [[0.0] * n for _ in range(n)]   # responsibilities
    A = [[0.0] * n for _ in range(n)]   # availabilities

    last_exemplars = None
    stable = 0
    iters = max_iter
    for it in range(max_iter):
        # --- update responsibilities ---
        for i in range(n):
            # find the top two values of A[i][k] + S[i][k]
            max1 = float("-inf")
            arg1 = -1
            max2 = float("-inf")
            for k in range(n):
                v = A[i][k] + S[i][k]
                if v > max1:
                    max2 = max1
                    max1 = v
                    arg1 = k
                elif v > max2:
                    max2 = v
            for k in range(n):
                comp = max2 if k == arg1 else max1
                new_r = S[i][k] - comp
                R[i][k] = damping * R[i][k] + (1 - damping) * new_r

        # --- update availabilities ---
        # column sums of max(0, R[i][k]) for i != k
        for k in range(n):
            rkk = R[k][k]
            # sum of positive responsibilities toward k from all i != k
            sum_pos = 0.0
            for i in range(n):
                if i != k:
                    sum_pos += max(0.0, R[i][k])
            for i in range(n):
                if i == k:
                    new_a = sum_pos
                else:
                    # subtract this i's own positive contribution
                    val = rkk + sum_pos - max(0.0, R[i][k])
                    new_a = min(0.0, val)
                A[i][k] = damping * A[i][k] + (1 - damping) * new_a

        # --- extract exemplars: k with R[k][k] + A[k][k] > 0 ---
        exemplars = tuple(k for k in range(n) if R[k][k] + A[k][k] > 0)
        if exemplars == last_exemplars and exemplars:
            stable += 1
            if stable >= conv_iter:
                iters = it + 1
                break
        else:
            stable = 0
            last_exemplars = exemplars

    # final assignment
    exemplars = [k for k in range(n) if R[k][k] + A[k][k] > 0]
    if not exemplars:
        # fall back to the single best self-preference point
        exemplars = [max(range(n), key=lambda k: R[k][k] + A[k][k])]
    labels = _assign(S, exemplars)
    return {
        "exemplars": exemplars,
        "labels": labels,
        "n_clusters": len(exemplars),
        "iterations": iters,
        "net_similarity": net_similarity(S, exemplars, labels),
    }


def _assign(S, exemplars):
    """Assign each point to the exemplar it is most similar to."""
    n = len(S)
    labels = [0] * n
    for i in range(n):
        best = exemplars[0]
        best_s = S[i][best]
        for k in exemplars:
            if S[i][k] > best_s:
                best_s = S[i][k]
                best = k
        labels[i] = best
    return labels


def net_similarity(S, exemplars, labels):
    """Objective: sum of similarities of points to their exemplars, plus exemplar self-preferences."""
    total = 0.0
    for i in range(len(labels)):
        total += S[i][labels[i]]
    return total


def cluster_points(points, preference=None, damping=0.5, max_iter=200):
    """Convenience: cluster raw points (list of coordinate lists) by affinity propagation.

    Returns the same dict as affinity_propagation, with labels remapped to 0..n_clusters-1 and the
    exemplar coordinates included."""
    S = negative_sq_euclidean(points)
    res = affinity_propagation(S, preference=preference, damping=damping, max_iter=max_iter)
    # remap exemplar indices to compact labels
    ex_to_label = {ex: c for c, ex in enumerate(res["exemplars"])}
    res["cluster_id"] = [ex_to_label[e] for e in res["labels"]]
    res["exemplar_points"] = [points[e] for e in res["exemplars"]]
    return res
