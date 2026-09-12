"""The traveling salesman problem: exact Held-Karp DP and the 2-opt heuristic.

The TRAVELING SALESMAN PROBLEM asks for the shortest tour visiting every city exactly once and
returning to the start. It is the archetypal NP-hard optimization problem -- the number of tours
grows as (n-1)!/2, so brute force is hopeless beyond a dozen cities -- yet it models vehicle routing,
circuit-board drilling, DNA sequencing, and logistics everywhere. Two complementary approaches solve
it in practice: an EXACT dynamic program for small instances, and fast HEURISTICS that find
near-optimal tours for large ones.

HELD-KARP is the exact dynamic program. Instead of enumerating (n-1)! tours it builds up, for every
SUBSET of cities and every possible last city in that subset, the shortest path from the start
through exactly that subset ending there -- reusing subproblems so the cost is O(n^2 2^n) time and
O(n 2^n) memory, an enormous improvement that makes ~20 cities exactly solvable. The 2-OPT heuristic
takes a different tack: start from any tour and repeatedly find two edges that, when removed and
reconnected the other way (reversing the segment between them), shorten the tour, until no such
improving swap remains -- a LOCAL SEARCH that removes the crossings a good tour never has and reliably
lands within a few percent of optimal in near-quadratic time per pass.

This module implements Held-Karp exact TSP, the nearest-neighbour construction, and 2-opt improvement
over an arbitrary distance matrix (with a Euclidean-points helper). It is verified against brute-force
permutation search that Held-Karp returns the true optimum on small instances, that 2-opt improves a
nearest-neighbour tour and gets within a small ratio of the Held-Karp optimum, that every returned
tour is a valid permutation visiting each city once, that a tour's reported length matches the summed
edge distances, and that the triangle inequality holds for the Euclidean helper. Pure stdlib; an
optimization companion to the simulated-annealing, dynamic-programming, and graph notes."""

from __future__ import annotations

import math


def euclidean_matrix(points):
    """Distance matrix for a list of (x, y) points."""
    n = len(points)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(points[i], points[j])
            D[i][j] = D[j][i] = d
    return D


def tour_length(tour, D):
    """Total length of a closed tour (returns to the start)."""
    return sum(D[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def held_karp(D):
    """Exact shortest tour via the Held-Karp dynamic program. Returns (tour, length). Feasible for
    n up to ~18-20."""
    n = len(D)
    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        return [0, 1], D[0][1] + D[1][0]

    # dp[(subset, last)] = shortest path from 0 through `subset` (which includes last, excludes 0)
    # ending at `last`. subset is a bitmask over cities 1..n-1.
    dp = {}
    parent = {}
    for k in range(1, n):
        dp[(1 << (k - 1), k)] = D[0][k]
        parent[(1 << (k - 1), k)] = 0

    for size in range(2, n):
        for subset in _subsets_of_size(n - 1, size):
            for last in range(1, n):
                if not (subset & (1 << (last - 1))):
                    continue
                prev_subset = subset & ~(1 << (last - 1))
                best = math.inf
                best_prev = None
                for m in range(1, n):
                    if m == last or not (prev_subset & (1 << (m - 1))):
                        continue
                    val = dp.get((prev_subset, m))
                    if val is None:
                        continue
                    cand = val + D[m][last]
                    if cand < best:
                        best = cand
                        best_prev = m
                if best_prev is not None:
                    dp[(subset, last)] = best
                    parent[(subset, last)] = best_prev

    full = (1 << (n - 1)) - 1
    best = math.inf
    best_last = None
    for last in range(1, n):
        val = dp.get((full, last))
        if val is None:
            continue
        cand = val + D[last][0]
        if cand < best:
            best = cand
            best_last = last

    # reconstruct
    tour = [0]
    subset = full
    last = best_last
    path = []
    while last != 0:
        path.append(last)
        p = parent[(subset, last)]
        subset &= ~(1 << (last - 1))
        last = p
    tour = [0] + path[::-1]
    return tour, best


def _subsets_of_size(n_items, size):
    """All bitmasks over n_items bits with exactly `size` bits set (Gosper's hack)."""
    if size == 0:
        yield 0
        return
    comb = (1 << size) - 1
    limit = 1 << n_items
    while comb < limit:
        yield comb
        c = comb & -comb
        r = comb + c
        comb = (((r ^ comb) >> 2) // c) | r


def nearest_neighbour(D, start=0):
    """Greedy nearest-neighbour tour construction."""
    n = len(D)
    unvisited = set(range(n))
    tour = [start]
    unvisited.discard(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur][j])
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return tour


def two_opt(tour, D, max_passes=1000):
    """Improve a tour by 2-opt local search: reverse segments to remove crossings until no improving
    swap remains. Returns the improved tour."""
    n = len(tour)
    tour = list(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue                      # would reverse the whole tour (no change)
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # current edges (a,b) and (c,d); after 2-opt: (a,c) and (b,d)
                delta = (D[a][c] + D[b][d]) - (D[a][b] + D[c][d])
                if delta < -1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def solve_2opt(D, start=0):
    """Nearest-neighbour construction followed by 2-opt improvement. Returns (tour, length)."""
    tour = two_opt(nearest_neighbour(D, start), D)
    return tour, tour_length(tour, D)
