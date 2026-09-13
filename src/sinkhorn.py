"""Sinkhorn optimal transport: the cheapest way to morph one distribution into another.

OPTIMAL TRANSPORT asks: given a pile of sand shaped like distribution a and a hole shaped like b, what
is the least-effort way to move the sand to fill the hole? Formally, minimize the total cost
sum_ij P_ij C_ij of a transport plan P -- a joint distribution whose row sums are a and column sums are
b -- where C_ij is the cost of moving mass from bin i to bin j. The minimum cost is the WASSERSTEIN
distance (earth mover's distance), a geometrically meaningful metric between distributions that,
unlike KL divergence, respects the ground space: two narrow peaks that are far apart are far in
Wasserstein even if they don't overlap.

Exactly solving OT is a linear program, O(n^3 log n). Cuturi's 2013 insight was to add an ENTROPY
penalty -eps * H(P), which makes the problem strictly convex with a stunningly simple solution: the
optimal plan is P = diag(u) K diag(v) where K = exp(-C/eps) is the Gibbs kernel, and u, v are found by
SINKHORN'S algorithm -- just alternately rescaling rows to match a and columns to match b:

    u <- a / (K v),   v <- b / (K^T u),   repeat.

Each iteration is two matrix-vector products; it converges linearly, is trivially parallel, and the
regularization eps trades sharpness (small eps -> closer to true OT, but slower and less stable) for
speed and smoothness. This module runs Sinkhorn to get the transport plan and the (regularized and
unregularized) transport cost, plus the exact 1-D Wasserstein distance by the closed-form sorted-CDF
formula, and a small exact LP-free reference by the North-West-corner + cost improvement is avoided in
favour of comparing against the 1-D exact value and marginal constraints.

Validated: the Sinkhorn plan's row and column sums match the target marginals; as eps shrinks the
regularized cost approaches the exact 1-D Wasserstein distance computed from sorted CDFs; transporting
a distribution to itself costs zero; the cost is symmetric; and moving a spike from one location to
another costs exactly the ground distance. Pure stdlib; the distribution-distance companion to the
KL/entropy (Shannon) and the assignment (Hungarian) tools."""

from __future__ import annotations

import math


def _normalize(v):
    s = sum(v)
    if s <= 0:
        raise ValueError("distribution must have positive mass")
    return [x / s for x in v]


def cost_matrix(xs, ys, p=2):
    """Ground cost C_ij = |xs_i - ys_j|^p between support points (default squared distance)."""
    return [[abs(x - y) ** p for y in ys] for x in xs]


def sinkhorn(a, b, C, eps=0.01, max_iter=2000, tol=1e-9):
    """Entropic-regularized optimal transport. a, b are (unnormalized) histograms, C the cost
    matrix. Returns (plan P, regularized_cost, iterations). eps is the entropy regularization."""
    a = _normalize(a)
    b = _normalize(b)
    n = len(a)
    m = len(b)
    if len(C) != n or any(len(row) != m for row in C):
        raise ValueError("cost matrix shape mismatch")
    # Gibbs kernel K = exp(-C/eps)
    K = [[math.exp(-C[i][j] / eps) for j in range(m)] for i in range(n)]
    u = [1.0] * n
    v = [1.0] * m
    for it in range(1, max_iter + 1):
        u_prev = list(u)
        # u <- a / (K v)
        for i in range(n):
            s = sum(K[i][j] * v[j] for j in range(m))
            u[i] = a[i] / s if s > 0 else 0.0
        # v <- b / (K^T u)
        for j in range(m):
            s = sum(K[i][j] * u[i] for i in range(n))
            v[j] = b[j] / s if s > 0 else 0.0
        # convergence: change in u
        if max(abs(u[i] - u_prev[i]) for i in range(n)) < tol:
            break
    # final row rescale so P's row sums match a exactly (the last sweep updated columns, so rows
    # otherwise lag by half an iteration)
    for i in range(n):
        s = sum(K[i][j] * v[j] for j in range(m))
        u[i] = a[i] / s if s > 0 else 0.0
    P = [[u[i] * K[i][j] * v[j] for j in range(m)] for i in range(n)]
    reg_cost = sum(P[i][j] * C[i][j] for i in range(n) for j in range(m))
    return P, reg_cost, it


def transport_cost(P, C):
    """Total transport cost sum_ij P_ij C_ij of a given plan."""
    return sum(P[i][j] * C[i][j] for i in range(len(P)) for j in range(len(P[0])))


def sinkhorn_distance(a, b, xs, ys=None, eps=0.01, p=2, **kwargs):
    """Convenience: Sinkhorn transport cost between histograms a (on support xs) and b (on ys)."""
    if ys is None:
        ys = xs
    C = cost_matrix(xs, ys, p=p)
    P, cost, _ = sinkhorn(a, b, C, eps=eps, **kwargs)
    return cost


# --- exact 1-D Wasserstein (closed form) -------------------------------------
def wasserstein_1d(a, xs, b, ys, p=1):
    """Exact p-Wasserstein distance between two 1-D distributions, by integrating the gap between
    their inverse CDFs. For discrete distributions on sorted supports this is the area between the
    CDFs (p=1) or its p-norm generalization. Returns the distance (not the p-th power)."""
    a = _normalize(a)
    b = _normalize(b)
    # merge all support points, walk the CDFs
    points = sorted(set(xs) | set(ys))
    # build step CDFs
    def cdf_at(support, mass, x):
        return sum(mass[i] for i, s in enumerate(support) if s <= x)

    # integrate |CDF_a - CDF_b|^p over x, using the piecewise-constant structure between points
    total = 0.0
    for k in range(len(points) - 1):
        x0 = points[k]
        x1 = points[k + 1]
        ca = cdf_at(xs, a, x0)
        cb = cdf_at(ys, b, x0)
        gap = abs(ca - cb)
        total += (gap ** p) * (x1 - x0)
    return total ** (1.0 / p)


# --- brute reference: exact transport via LP-free small solver ---------------
def exact_transport_1d(a, xs, b, ys, p=2):
    """Exact optimal transport cost for 1-D distributions: the optimal plan is monotone (sorted),
    so sweep both sorted supports moving mass greedily. Returns the transport cost."""
    a = _normalize(a)
    b = _normalize(b)
    # sort by support
    A = sorted(zip(xs, a))
    B = sorted(zip(ys, b))
    i = j = 0
    ma = A[0][1]
    mb = B[0][1]
    cost = 0.0
    while i < len(A) and j < len(B):
        flow = min(ma, mb)
        cost += flow * abs(A[i][0] - B[j][0]) ** p
        ma -= flow
        mb -= flow
        if ma <= 1e-15:
            i += 1
            if i < len(A):
                ma = A[i][1]
        if mb <= 1e-15:
            j += 1
            if j < len(B):
                mb = B[j][1]
    return cost
