"""Ant colony optimization: solving the travelling salesman with simulated pheromones.

Real ants find short paths to food without a map: each lays a PHEROMONE trail, shorter paths get
traversed faster so their trails are reinforced sooner, and other ants prefer strongly-scented
edges -- a positive feedback that converges the colony onto good routes. Ant colony optimization
(Dorigo, 1992) turns that into an algorithm for hard combinatorial problems, the travelling
salesman being the canonical one.

Each iteration a swarm of artificial ants each builds a tour. At every step an ant at city i picks
the next city j with probability proportional to

    tau_ij^alpha  *  eta_ij^beta

where tau_ij is the PHEROMONE on edge (i,j) (learned, shared memory) and eta_ij = 1/distance is the
HEURISTIC desirability (greedy, local). alpha weights experience, beta weights greed. After all
ants finish, pheromone EVAPORATES (tau <- (1-rho) tau, forgetting stale trails) and each ant
DEPOSITS pheromone on its edges inversely proportional to its tour length, so shorter tours leave
stronger trails. Over many iterations the pheromone concentrates on the edges of good tours and the
colony converges.

This module implements ant system / ant colony optimization for the symmetric TSP with the standard
transition rule, evaporation, and length-weighted deposit (plus optional elitist reinforcement of
the best-so-far tour) -- verified that it recovers the optimal perimeter of a square and of points
on a circle, that it beats the nearest-neighbour greedy tour on random cities, that the best tour
length decreases over iterations, and that pheromone concentrates on short edges. Pure stdlib (its
own RNG); a swarm-intelligence companion to the simulated-annealing and genetic-algorithm notes."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed=0):
        self.state = seed & 0xFFFFFFFF

    def uniform(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)     # high bits


def tour_length(order, dist):
    n = len(order)
    return sum(dist[order[i]][order[(i + 1) % n]] for i in range(n))


def _distance_matrix(points):
    n = len(points)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            D[i][j] = d
            D[j][i] = d
    return D


def nearest_neighbour_tour(dist, start=0):
    """Greedy baseline: always hop to the nearest unvisited city."""
    n = len(dist)
    unvisited = set(range(n))
    order = [start]
    unvisited.discard(start)
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda c: dist[last][c])
        order.append(nxt)
        unvisited.discard(nxt)
    return order


def solve(points, n_ants=None, n_iter=100, alpha=1.0, beta=3.0, rho=0.5, q=1.0,
          elitist=True, seed=0, track=False):
    """Solve a symmetric TSP by ant colony optimization.

    points  : list of (x, y) cities.
    alpha   : pheromone weight; beta : heuristic (1/distance) weight; rho : evaporation rate.
    q       : deposit constant; elitist reinforces the best-so-far tour each round.
    Returns (best_order, best_length) or, with track=True, also a history dict."""
    n = len(points)
    if n <= 2:
        return list(range(n)), tour_length(list(range(n)), _distance_matrix(points)) if n else 0.0
    dist = _distance_matrix(points)
    rng = _Rng(seed)
    if n_ants is None:
        n_ants = n

    # heuristic desirability eta = 1/distance (0 on the diagonal)
    eta = [[0.0 if i == j else 1.0 / dist[i][j] for j in range(n)] for i in range(n)]
    # initialize pheromone uniformly, scaled to a greedy tour length (standard heuristic)
    nn_len = tour_length(nearest_neighbour_tour(dist), dist)
    tau0 = 1.0 / (n * nn_len) if nn_len > 0 else 1.0
    tau = [[tau0] * n for _ in range(n)]

    best_order = None
    best_len = float("inf")
    history = []

    for _ in range(n_iter):
        all_tours = []
        for _ in range(n_ants):
            order = _build_tour(n, tau, eta, alpha, beta, rng)
            length = tour_length(order, dist)
            all_tours.append((order, length))
            if length < best_len:
                best_len, best_order = length, list(order)

        # EVAPORATE
        for i in range(n):
            for j in range(n):
                tau[i][j] *= (1.0 - rho)

        # DEPOSIT: each ant lays q/length on its edges
        for order, length in all_tours:
            amount = q / length
            for k in range(n):
                a, b = order[k], order[(k + 1) % n]
                tau[a][b] += amount
                tau[b][a] += amount

        # ELITIST: extra reinforcement of the best-so-far tour
        if elitist and best_order is not None:
            amount = q / best_len
            for k in range(n):
                a, b = best_order[k], best_order[(k + 1) % n]
                tau[a][b] += amount
                tau[b][a] += amount

        history.append(best_len)

    if track:
        return best_order, best_len, {"history": history, "pheromone": tau, "dist": dist}
    return best_order, best_len


def _build_tour(n, tau, eta, alpha, beta, rng):
    """One ant constructs a tour by probabilistic next-city choices."""
    start = int(rng.uniform() * n) % n
    visited = [False] * n
    visited[start] = True
    order = [start]
    for _ in range(n - 1):
        cur = order[-1]
        # weight each unvisited city by tau^alpha * eta^beta
        weights = []
        total = 0.0
        for j in range(n):
            if visited[j]:
                weights.append(0.0)
                continue
            w = (tau[cur][j] ** alpha) * (eta[cur][j] ** beta)
            weights.append(w)
            total += w
        # roulette-wheel selection
        if total <= 0:
            nxt = next(j for j in range(n) if not visited[j])
        else:
            r = rng.uniform() * total
            acc = 0.0
            nxt = None
            for j in range(n):
                acc += weights[j]
                if acc >= r and not visited[j]:
                    nxt = j
                    break
            if nxt is None:
                nxt = next(j for j in range(n) if not visited[j])
        visited[nxt] = True
        order.append(nxt)
    return order
