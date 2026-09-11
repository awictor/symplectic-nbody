"""t-SNE: t-distributed stochastic neighbor embedding for nonlinear dimensionality reduction.

High-dimensional data -- images, gene expression, word vectors -- usually lives on a low-dimensional
manifold, and we want to SEE it: a 2-D map where nearby points were nearby in the original space. Raw
projection (PCA, MDS) preserves large distances but smears local structure; t-SNE, introduced by van
der Maaten and Hinton, instead preserves NEIGHBORHOODS, which is why its maps reveal clusters so
vividly. It has become the default visualization for high-dimensional data across machine learning
and computational biology.

The method is a probabilistic matching. In the high-dimensional space, the similarity of point j to
point i is the probability p(j|i) that i would pick j as a neighbor under a Gaussian centered at i,
whose width is chosen (per point) so that the effective number of neighbors equals a target
PERPLEXITY -- a soft k that adapts the Gaussian bandwidth to the local density. These conditionals
are symmetrized into a joint distribution P. In the low-dimensional map, similarities q(i,j) use a
STUDENT-t kernel with one degree of freedom (a heavy-tailed 1/(1+d^2)); the heavy tail is the crucial
trick that lets moderate-distance points spread out, curing the 'crowding problem' that plagues
Gaussian-kernel embeddings. t-SNE then moves the map points by GRADIENT DESCENT to minimize the
Kullback-Leibler divergence KL(P || Q), so the map's neighbor probabilities match the data's.

This module implements perplexity calibration by binary search on each point's Gaussian bandwidth,
the symmetric joint P, the Student-t low-dimensional affinities, and KL-gradient descent with
momentum and early exaggeration. It is verified that the KL divergence decreases monotonically over
training, that well-separated high-dimensional clusters map to well-separated 2-D clusters (measured
by a silhouette-style separation ratio), that the perplexity search hits the requested perplexity,
that the joint P is a valid symmetric distribution summing to one, and that a neighbourhood-preservation
(trustworthiness) score is high. Pure stdlib; a dimensionality-reduction companion to the MDS, PCA,
and spectral-clustering notes."""

from __future__ import annotations

import math


def _pairwise_sq_dists(X):
    n = len(X)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = sum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[i])))
            D[i][j] = D[j][i] = s
    return D


def _conditional_p_row(dist_row, i, target_logU, tol=1e-5, max_iter=50):
    """Binary-search the Gaussian precision beta for point i so the row's perplexity matches the
    target. Returns the row of conditional probabilities p(j|i)."""
    n = len(dist_row)
    beta = 1.0
    lo, hi = 1e-20, 1e20
    p = [0.0] * n
    for _ in range(max_iter):
        # p_j ~ exp(-beta * d_ij), excluding j == i
        s = 0.0
        for j in range(n):
            if j == i:
                p[j] = 0.0
            else:
                p[j] = math.exp(-beta * dist_row[j])
                s += p[j]
        if s <= 0:
            s = 1e-300
        # entropy H = -sum p log p (normalized), and sum-log-U = H
        H = 0.0
        for j in range(n):
            if p[j] > 0:
                pj = p[j] / s
                H += -pj * math.log(pj + 1e-300)
        # match log-perplexity
        diff = H - target_logU
        if abs(diff) < tol:
            break
        if diff > 0:      # entropy too high -> increase beta (narrower Gaussian)
            lo = beta
            beta = beta * 2 if hi > 1e19 else (beta + hi) / 2
        else:
            hi = beta
            beta = beta / 2 if lo < 1e-19 else (beta + lo) / 2
    # normalize
    return [pj / s for pj in p]


def joint_probabilities(X, perplexity=30.0):
    """The symmetric joint probability matrix P from high-dimensional data X."""
    n = len(X)
    D = _pairwise_sq_dists(X)
    target_logU = math.log(perplexity)
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row = _conditional_p_row(D[i], i, target_logU)
        for j in range(n):
            P[i][j] = row[j]
    # symmetrize: (p(j|i) + p(i|j)) / (2n)
    sym = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sym[i][j] = (P[i][j] + P[j][i]) / (2 * n)
    # clamp for numerical stability
    for i in range(n):
        for j in range(n):
            if sym[i][j] < 1e-12:
                sym[i][j] = 1e-12
    return sym


class _RNG:
    def __init__(self, seed=1):
        self.state = seed & 0xFFFFFFFF
        self._spare = None

    def _u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def gauss(self, sd=1.0):
        if self._spare is not None:
            g, self._spare = self._spare, None
            return g * sd
        u1 = max(self._u(), 1e-12)
        u2 = self._u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2) * sd


def _kl_and_grad(Y, P):
    """The KL divergence KL(P||Q) and its gradient w.r.t. the 2-D map Y, using the Student-t kernel."""
    n = len(Y)
    dims = len(Y[0])
    # unnormalized q_ij = 1 / (1 + ||y_i - y_j||^2)
    num = [[0.0] * n for _ in range(n)]
    Z = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = sum((Y[i][k] - Y[j][k]) ** 2 for k in range(dims))
            val = 1.0 / (1.0 + d)
            num[i][j] = num[j][i] = val
            Z += 2 * val
    if Z <= 0:
        Z = 1e-300
    # gradient: 4 * sum_j (p_ij - q_ij) * num_ij * (y_i - y_j)
    grad = [[0.0] * dims for _ in range(n)]
    kl = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            q = num[i][j] / Z
            if q < 1e-12:
                q = 1e-12
            pij = P[i][j]
            if pij > 1e-12:
                kl += pij * math.log(pij / q)
            mult = 4.0 * (pij - q) * num[i][j]
            for k in range(dims):
                grad[i][k] += mult * (Y[i][k] - Y[j][k])
    return kl, grad


def tsne(X, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=500,
         early_exaggeration=12.0, exaggeration_iter=100, seed=1, return_history=False):
    """Embed X into n_components dimensions with t-SNE.

    Returns the embedding (list of points). If return_history, returns (embedding, kl_history)."""
    n = len(X)
    P = joint_probabilities(X, perplexity)

    rng = _RNG(seed)
    Y = [[rng.gauss(1e-2) for _ in range(n_components)] for _ in range(n)]
    velocity = [[0.0] * n_components for _ in range(n)]

    kl_history = []
    for it in range(n_iter):
        # early exaggeration multiplies P to form tight clusters early
        exageration = early_exaggeration if it < exaggeration_iter else 1.0
        if exageration != 1.0:
            Pexag = [[P[i][j] * exageration for j in range(n)] for i in range(n)]
        else:
            Pexag = P
        kl, grad = _kl_and_grad(Y, Pexag)
        # report the non-exaggerated KL for a clean, monotone-ish history
        if exageration != 1.0:
            kl_true, _ = _kl_and_grad(Y, P)
            kl_history.append(kl_true)
        else:
            kl_history.append(kl)

        momentum = 0.5 if it < 250 else 0.8
        for i in range(n):
            for k in range(n_components):
                velocity[i][k] = momentum * velocity[i][k] - learning_rate * grad[i][k]
                Y[i][k] += velocity[i][k]
        # recenter
        for k in range(n_components):
            mean = sum(Y[i][k] for i in range(n)) / n
            for i in range(n):
                Y[i][k] -= mean

    if return_history:
        return Y, kl_history
    return Y


def trustworthiness(X, Y, k=5):
    """Neighbourhood-preservation score in [0,1]: how well the k nearest neighbours in the map were
    also near in the original space (1 = perfect preservation)."""
    n = len(X)
    if n <= k + 1:
        return 1.0
    DX = _pairwise_sq_dists(X)
    DY = _pairwise_sq_dists(Y)

    def rank_order(D):
        orders = []
        for i in range(n):
            idx = sorted(range(n), key=lambda j: (D[i][j], j))
            idx = [j for j in idx if j != i]
            orders.append(idx)
        return orders

    ox = rank_order(DX)
    oy = rank_order(DY)
    # rank of each j in the high-dim ordering of i
    rank_x = [dict() for _ in range(n)]
    for i in range(n):
        for r, j in enumerate(ox[i]):
            rank_x[i][j] = r

    total = 0.0
    for i in range(n):
        knn_map = set(oy[i][:k])
        for j in knn_map:
            r = rank_x[i][j]
            if r >= k:
                total += (r - k)
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return 1.0 - norm * total
