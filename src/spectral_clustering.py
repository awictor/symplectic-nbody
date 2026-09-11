"""Spectral clustering: cutting a graph by the eigenvectors of its Laplacian.

k-means fails on non-convex shapes -- two interlocking moons, concentric rings -- because it splits
space by distance to a centroid. Spectral clustering escapes that trap by working on a GRAPH
instead of raw coordinates: connect nearby points with weighted edges, then cut the graph into
pieces that are densely connected inside and sparsely connected between. The magic is that this
combinatorial cut is solved (in relaxed form) by LINEAR ALGEBRA -- the eigenvectors of the graph
LAPLACIAN.

The recipe (Ng, Jordan, Weiss, 2002):

  1. Build an AFFINITY matrix W, w_ij = exp(-||x_i - x_j||^2 / 2 sigma^2): nearby points strongly
     connected, distant points nearly disconnected.
  2. Form the graph Laplacian L = D - W (D the diagonal degree matrix), and its normalized form.
  3. Take the eigenvectors of the k SMALLEST eigenvalues -- they encode the cluster structure. The
     number of eigenvalues near zero equals the number of connected components (a theorem).
  4. Embed each point by its coordinates in those eigenvectors and run k-means in that space.

In the eigenvector embedding the tangled moons become two tight, linearly separable blobs, so
k-means -- which failed on the original coordinates -- succeeds trivially. This module builds the
affinity graph, the unnormalized and symmetric-normalized Laplacians, extracts the low
eigenvectors (via the reuse of a symmetric eigensolver on cI - L, turning smallest into largest),
and clusters the embedding with k-means -- verified that it separates two concentric rings and two
moons that k-means cannot, that the Laplacian's zero-eigenvalue multiplicity counts connected
components, and that it recovers plain blobs too. Pure stdlib, built on the eigen and k-means
modules; the graph-based companion to the DBSCAN and k-means clustering notes."""

from __future__ import annotations

import math

from eigen import eigenvalues_symmetric
from kmeans import kmeans_best


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def affinity_matrix(X, sigma=1.0):
    """Gaussian (RBF) affinity W, w_ij = exp(-||x_i - x_j||^2 / 2 sigma^2), zero self-affinity."""
    n = len(X)
    W = [[0.0] * n for _ in range(n)]
    denom = 2.0 * sigma * sigma
    for i in range(n):
        for j in range(i + 1, n):
            w = math.exp(-_dist2(X[i], X[j]) / denom)
            W[i][j] = w
            W[j][i] = w
    return W


def knn_affinity(X, k=10, sigma=1.0):
    """A sparser affinity: keep each point's k nearest neighbours (symmetrized). Often cleaner than
    the full Gaussian graph for well-separated manifolds."""
    n = len(X)
    denom = 2.0 * sigma * sigma
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        d = sorted(range(n), key=lambda j: _dist2(X[i], X[j]))
        for j in d[1:k + 1]:
            w = math.exp(-_dist2(X[i], X[j]) / denom)
            W[i][j] = max(W[i][j], w)
            W[j][i] = max(W[j][i], w)
    return W


def laplacian(W, normalized=True):
    """Graph Laplacian. Unnormalized L = D - W; symmetric-normalized L_sym = I - D^-1/2 W D^-1/2."""
    n = len(W)
    deg = [sum(W[i]) for i in range(n)]
    if not normalized:
        return [[(deg[i] if i == j else 0.0) - W[i][j] for j in range(n)] for i in range(n)]
    dinv = [1.0 / math.sqrt(deg[i]) if deg[i] > 1e-12 else 0.0 for i in range(n)]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            L[i][j] = (1.0 if i == j else 0.0) - dinv[i] * W[i][j] * dinv[j]
    return L


def smallest_eigenvectors(L, k):
    """The k eigenvectors of L with the SMALLEST eigenvalues.

    eigenvalues_symmetric returns largest-first, so we solve on (cI - L): its largest eigenvalues
    are c minus the smallest of L, with the same eigenvectors."""
    n = len(L)
    # Gershgorin bound guarantees c >= max eigenvalue of L
    c = max(sum(abs(L[i][j]) for j in range(n)) for i in range(n)) + 1.0
    M = [[(c if i == j else 0.0) - L[i][j] for j in range(n)] for i in range(n)]
    vals, vecs = eigenvalues_symmetric(M)
    # largest vals of M correspond to smallest of L; take the top k eigenvectors
    order = sorted(range(n), key=lambda t: -vals[t])
    chosen = [vecs[order[t]] for t in range(k)]
    small_vals = [c - vals[order[t]] for t in range(k)]      # recover L's eigenvalues
    return small_vals, chosen


def spectral_clustering(X, n_clusters, sigma=1.0, affinity="rbf", knn=10,
                        normalized=True, restarts=8, seed=1):
    """Cluster X into n_clusters groups by the spectrum of the affinity-graph Laplacian.

    Returns (labels, eigenvalues) where eigenvalues are the n_clusters smallest of the Laplacian
    (their near-zero count indicates the natural number of components)."""
    if affinity == "knn":
        W = knn_affinity(X, k=knn, sigma=sigma)
    else:
        W = affinity_matrix(X, sigma=sigma)
    L = laplacian(W, normalized=normalized)
    vals, vecs = smallest_eigenvectors(L, n_clusters)
    # embedding: row i is (vec_0[i], vec_1[i], ..., vec_{k-1}[i])
    n = len(X)
    embedding = [[vecs[c][i] for c in range(n_clusters)] for i in range(n)]
    # row-normalize (the Ng-Jordan-Weiss step) so points lie on a sphere
    emb = []
    for row in embedding:
        norm = math.sqrt(sum(v * v for v in row)) or 1.0
        emb.append([v / norm for v in row])
    _, labels, _ = kmeans_best(emb, n_clusters, restarts=restarts, seed=seed)
    return labels, vals


def count_components(W, tol=1e-6):
    """Number of connected components of the affinity graph = multiplicity of Laplacian eigenvalue
    0. Computed directly by union-find on the nonzero edges (a check on the spectral count)."""
    n = len(W)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if W[i][j] > tol:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})
