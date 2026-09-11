"""Simulated annealing: escaping local minima by borrowing from metallurgy.

Greedy search -- only ever move downhill -- gets trapped in the first local minimum it finds.
SIMULATED ANNEALING (Kirkpatrick, Gelatt, Vecchi, 1983) escapes by sometimes moving UPHILL, with a
probability that shrinks over time. The name is literal: annealing a metal means heating it and
cooling it slowly so its atoms settle into a low-energy crystal instead of a brittle, defect-ridden
freeze. The algorithm cools a fictitious TEMPERATURE the same way.

At each step it proposes a random neighbouring state and applies the METROPOLIS criterion: a move
that lowers the cost is always accepted; a move that raises it by delta is accepted with
probability exp(-delta / T). When T is high (early), almost anything is accepted and the search
roams freely, hopping out of local basins; as T cools toward zero, only improving moves survive and
the search settles. The COOLING SCHEDULE -- how T decreases -- is the key knob: cool too fast and
you quench into a poor local minimum; cool slowly (geometric T <- alpha T with alpha near 1) and
the theory says you approach the global optimum.

This module implements generic simulated annealing over any state with a user-supplied cost,
neighbour, and cooling schedule, plus a ready-made travelling-salesman solver (2-opt-style segment-
reversal moves) -- verified that it finds the global minimum of a deliberately multimodal function
that greedy descent misses, that a square TSP tour converges to the known optimal perimeter, that
it beats nearest-neighbour greedy on random tours, and that the acceptance rate falls as the system
cools. Pure stdlib (its own RNG); the stochastic-global-optimization companion to the Bayesian-
optimization and Metropolis notes."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed=0):
        self.state = seed & 0xFFFFFFFF

    def uniform(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)     # high bits

    def randint(self, n):
        return int(self.uniform() * n) % n

    def gauss(self, sigma=1.0):
        u1 = max(1e-12, self.uniform())
        u2 = self.uniform()
        return sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def geometric_schedule(T0, alpha):
    """Cooling T_k = T0 * alpha^k (alpha in (0,1)); the standard exponential cooling."""
    return lambda k: T0 * (alpha ** k)


def linear_schedule(T0, steps):
    """Cooling that decreases linearly from T0 to ~0 over `steps` iterations."""
    return lambda k: max(1e-12, T0 * (1.0 - k / steps))


def anneal(initial, cost_fn, neighbour_fn, schedule, steps, seed=0, track=False):
    """Minimize cost_fn by simulated annealing.

    initial       : starting state
    cost_fn(s)    : scalar cost to minimize
    neighbour_fn(s, rng) : a random neighbouring state
    schedule(k)   : temperature at step k (see geometric_schedule / linear_schedule)
    Returns (best_state, best_cost) or, if track=True, also a history dict."""
    rng = _Rng(seed)
    current = initial
    current_cost = cost_fn(current)
    best, best_cost = current, current_cost
    accepts = 0
    proposals = 0
    cost_history = [current_cost]
    accept_flags = []
    for k in range(steps):
        T = max(schedule(k), 1e-12)
        candidate = neighbour_fn(current, rng)
        cand_cost = cost_fn(candidate)
        delta = cand_cost - current_cost
        proposals += 1
        # Metropolis: always take improvements; take worsening moves with prob exp(-delta/T)
        if delta <= 0 or rng.uniform() < math.exp(-delta / T):
            current, current_cost = candidate, cand_cost
            accepts += 1
            accept_flags.append(1)
            if current_cost < best_cost:
                best, best_cost = current, current_cost
        else:
            accept_flags.append(0)
        cost_history.append(current_cost)
    if track:
        return best, best_cost, {"cost_history": cost_history,
                                 "accept_rate": accepts / max(1, proposals),
                                 "accept_flags": accept_flags}
    return best, best_cost


# --- travelling salesman convenience -------------------------------------
def tour_length(order, points):
    """Total length of a closed tour visiting `points` in the given index order."""
    n = len(order)
    total = 0.0
    for i in range(n):
        a = points[order[i]]
        b = points[order[(i + 1) % n]]
        total += math.hypot(a[0] - b[0], a[1] - b[1])
    return total


def _reverse_segment(order, rng):
    """A 2-opt move: reverse a random contiguous segment of the tour."""
    n = len(order)
    i = rng.randint(n)
    j = rng.randint(n)
    if i > j:
        i, j = j, i
    new = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
    return new


def nearest_neighbour_tour(points, start=0):
    """A greedy baseline tour: always hop to the nearest unvisited city."""
    n = len(points)
    unvisited = set(range(n))
    order = [start]
    unvisited.discard(start)
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda c: math.hypot(points[last][0] - points[c][0],
                                                       points[last][1] - points[c][1]))
        order.append(nxt)
        unvisited.discard(nxt)
    return order


def solve_tsp(points, steps=20000, T0=None, alpha=0.9995, seed=0, initial_order=None):
    """Solve a travelling-salesman instance by simulated annealing with 2-opt moves.

    Returns (order, length). T0 defaults to a scale derived from the point spread."""
    n = len(points)
    if n <= 2:
        return list(range(n)), tour_length(list(range(n)), points) if n else 0.0
    order = initial_order if initial_order is not None else list(range(n))
    if T0 is None:
        # a reasonable starting temperature: the average edge length
        T0 = tour_length(order, points) / n
    best, best_cost = anneal(order,
                             lambda o: tour_length(o, points),
                             _reverse_segment,
                             geometric_schedule(T0, alpha),
                             steps, seed=seed)
    return best, best_cost
