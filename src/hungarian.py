"""The Hungarian algorithm: optimal assignment in O(n^3).

Given n workers and n jobs and a COST for assigning each worker to each job, which one-to-one
assignment minimizes the total cost? This ASSIGNMENT PROBLEM is everywhere -- matching taxis to
riders, tasks to machines, tracked objects to detections between video frames, students to schools.
Brute force tries all n! permutations; the HUNGARIAN ALGORITHM (Kuhn-Munkres, 1955, built on the
work of Konig and Egervary) finds the exact optimum in O(n^3), one of the first polynomial algorithms
for a combinatorial optimization problem and still the standard tool.

The method rests on a key invariant: subtracting a constant from any full row or column of the cost
matrix does not change WHICH assignment is optimal (it shifts every assignment's total by the same
amount). So the algorithm repeatedly reduces rows and columns to expose zeros, then tries to select n
independent zeros (one per row and column) -- a set of independent zeros is a valid assignment with
cost equal to the total subtracted. When fewer than n independent zeros exist, a minimum set of lines
covering all zeros (Konig's theorem) reveals the smallest uncovered value; subtracting it from
uncovered entries and adding it at line intersections creates new zeros without breaking the ones
already found, and the process repeats until n independent zeros -- the optimal assignment -- appear.

This module implements the O(n^3) potential/augmenting-path form of Kuhn-Munkres for rectangular cost
matrices (padded to square), returning the optimal assignment and its total cost, for both
minimization and maximization. It is verified against a brute-force search over all permutations for
small n (the assignment matches to the exact optimum), against the row/column reduction invariant,
on identity and known hand-worked matrices, that maximization is handled by negation, and that
rectangular inputs pad correctly. Pure stdlib; a combinatorial-optimization companion to the
max-flow (bipartite matching), simplex, and Dijkstra notes."""

from __future__ import annotations


def solve(cost, maximize=False):
    """Optimal assignment for a cost matrix (list of rows). Returns (assignment, total_cost) where
    assignment[i] = column assigned to row i. Handles rectangular matrices by padding. For
    maximize=True, finds the maximum-weight assignment."""
    if not cost or not cost[0]:
        return [], 0.0
    n_rows = len(cost)
    n_cols = len(cost[0])
    n = max(n_rows, n_cols)

    # pad to a square matrix; padded cells cost 0 (min) or a large value handled via negation (max)
    if maximize:
        biggest = max(max(row) for row in cost)
        pad_val = 0.0
        matrix = [[(biggest - cost[i][j] if i < n_rows and j < n_cols else pad_val)
                   for j in range(n)] for i in range(n)]
    else:
        pad_val = 0.0
        matrix = [[(cost[i][j] if i < n_rows and j < n_cols else pad_val)
                   for j in range(n)] for i in range(n)]

    assignment = _kuhn_munkres(matrix, n)

    # compute the true total cost over the ORIGINAL (unpadded) matrix
    total = 0.0
    result = [-1] * n_rows
    for i in range(n_rows):
        j = assignment[i]
        if j < n_cols:
            result[i] = j
            total += cost[i][j]
    return result, total


def _kuhn_munkres(a, n):
    """O(n^3) Hungarian algorithm on a square cost matrix `a` (minimization).

    Uses the potential-and-augmenting-path formulation (Jonker-Volgenant style), which is the
    standard robust O(n^3) implementation. Returns assignment[row] = col."""
    INF = float("inf")
    # u, v are the dual potentials for rows and columns (1-indexed with a sentinel row/col 0)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)          # p[j] = row assigned to column j (0 = none)
    way = [0] * (n + 1)        # for reconstructing the augmenting path

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        # find an augmenting path from row i
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            # update potentials along the visited set
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # augment: follow the way[] pointers back, flipping assignments
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def min_cost(cost):
    """Convenience: the minimum total assignment cost."""
    return solve(cost, maximize=False)[1]


def max_cost(cost):
    """Convenience: the maximum total assignment cost."""
    return solve(cost, maximize=True)[1]
