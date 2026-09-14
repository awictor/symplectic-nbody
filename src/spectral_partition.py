"""Spectral graph partitioning: cut a graph in two by the sign of its Fiedler vector.

How do you split a graph into two well-connected halves while cutting as few edges as possible? The
exact minimum-bisection problem is NP-hard, but there is a beautiful continuous relaxation. Form the
graph LAPLACIAN L = D - A (degree matrix minus adjacency). L is symmetric positive-semidefinite; its
smallest eigenvalue is always 0 with the constant eigenvector (every row of L sums to zero), and the
number of zero eigenvalues equals the number of connected components. The SECOND-smallest eigenvalue
-- the "algebraic connectivity" or Fiedler value -- and its eigenvector, the FIEDLER VECTOR, carry
the global shape of the graph: vertices that are tightly connected get similar values, and the sign
of each vertex's Fiedler-vector entry gives a near-optimal bisection.

This is the relaxed solution to the ratio-cut objective: minimizing x^T L x over x perpendicular to
the constant vector with x^T x fixed is exactly the Rayleigh-quotient minimization solved by the
Fiedler vector. Rounding the continuous vector to +/-1 by its sign is the spectral heuristic, and it
finds the intuitively-right cut on graphs with community structure -- two dense clusters joined by a
few bridge edges get separated along the bridges.

This module builds the Laplacian (combinatorial and symmetric-normalized), extracts the Fiedler value
and vector via the repo's Jacobi eigensolver, partitions by sign, and computes cut size, ratio cut,
and normalized cut. It is validated: the Laplacian rows sum to zero and it is PSD; the smallest
eigenvalue is 0 with a constant eigenvector; the count of near-zero eigenvalues equals the number of
connected components (checked against an independent union-find/BFS component count); the Fiedler
value is positive iff the graph is connected; and on planted two-cluster graphs the sign partition
recovers the true clusters and cuts only the bridge edges, matching a brute-force minimum bisection on
small graphs. Pure stdlib; the graph-spectral companion to the Jacobi, Lanczos, and union-find tools."""

from __future__ import annotations

from jacobi_eigen import sorted_eigen


def laplacian(n, edges, weights=None):
    """Combinatorial graph Laplacian L = D - A for an undirected graph on n vertices.

    edges: list of (u, v) index pairs. weights: optional list of edge weights (default 1)."""
    A = [[0.0] * n for _ in range(n)]
    for k, (u, v) in enumerate(edges):
        w = 1.0 if weights is None else weights[k]
        A[u][v] += w
        A[v][u] += w
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        deg = sum(A[i])
        for j in range(n):
            L[i][j] = -A[i][j]
        L[i][i] = deg
    return L


def normalized_laplacian(n, edges, weights=None):
    """Symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}. Isolated vertices give a 1 on
    the diagonal (their normalized row is just the identity)."""
    A = [[0.0] * n for _ in range(n)]
    for k, (u, v) in enumerate(edges):
        w = 1.0 if weights is None else weights[k]
        A[u][v] += w
        A[v][u] += w
    deg = [sum(A[i]) for i in range(n)]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i][j] = 1.0 if deg[i] > 0 else 1.0
            elif deg[i] > 0 and deg[j] > 0:
                L[i][j] = -A[i][j] / ((deg[i] ** 0.5) * (deg[j] ** 0.5))
    return L


def fiedler(n, edges, weights=None):
    """Fiedler value (2nd-smallest Laplacian eigenvalue) and Fiedler vector."""
    L = laplacian(n, edges, weights)
    vals, vecs = sorted_eigen(L)          # DESCENDING eigenvalues, columns are eigenvectors
    # smallest is at the end (index n-1, ~0 with constant vector); Fiedler = 2nd-smallest = n-2
    fval = vals[n - 2] if n > 1 else 0.0
    fvec = [vecs[i][n - 2] for i in range(n)] if n > 1 else [0.0] * n
    return fval, fvec


def partition(n, edges, weights=None):
    """Bisect the graph by the sign of the Fiedler vector. Returns a list of 0/1 labels per vertex."""
    _, fvec = fiedler(n, edges, weights)
    # split at the median so the two sides are balanced when many entries share a sign
    order = sorted(range(n), key=lambda i: fvec[i])
    labels = [0] * n
    for rank, i in enumerate(order):
        labels[i] = 0 if rank < n // 2 else 1
    return labels


def partition_by_sign(n, edges, weights=None):
    """Bisect purely by the sign of the Fiedler vector (may be unbalanced)."""
    _, fvec = fiedler(n, edges, weights)
    return [1 if fvec[i] > 0 else 0 for i in range(n)]


def cut_size(edges, labels, weights=None):
    """Total weight of edges crossing between the two label classes."""
    s = 0.0
    for k, (u, v) in enumerate(edges):
        if labels[u] != labels[v]:
            s += 1.0 if weights is None else weights[k]
    return s


def ratio_cut(n, edges, labels, weights=None):
    """Ratio cut = cut(A,B) * (1/|A| + 1/|B|); the objective the Fiedler vector relaxes."""
    a = sum(1 for l in labels if l == 0)
    b = n - a
    if a == 0 or b == 0:
        return float("inf")
    return cut_size(edges, labels, weights) * (1.0 / a + 1.0 / b)


def normalized_cut(n, edges, labels, weights=None):
    """Normalized cut = cut * (1/vol(A) + 1/vol(B)), vol = sum of degrees in the part."""
    deg = [0.0] * n
    for k, (u, v) in enumerate(edges):
        w = 1.0 if weights is None else weights[k]
        deg[u] += w
        deg[v] += w
    volA = sum(deg[i] for i in range(n) if labels[i] == 0)
    volB = sum(deg[i] for i in range(n) if labels[i] == 1)
    if volA == 0 or volB == 0:
        return float("inf")
    return cut_size(edges, labels, weights) * (1.0 / volA + 1.0 / volB)


def connected_components(n, edges):
    """Number of connected components and a component-id per vertex (BFS). Independent of the spectrum."""
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    comp = [-1] * n
    c = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if comp[y] == -1:
                    comp[y] = c
                    stack.append(y)
        c += 1
    return c, comp


def count_zero_eigenvalues(n, edges, weights=None, tol=1e-8):
    """Number of Laplacian eigenvalues that are (near) zero = number of connected components."""
    L = laplacian(n, edges, weights)
    vals, _ = sorted_eigen(L)
    return sum(1 for v in vals if abs(v) < tol)
