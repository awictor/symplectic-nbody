"""Tests for ant_colony: optimal tours, beats greedy, monotone best, pheromone concentration."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ant_colony import (solve, tour_length, nearest_neighbour_tour, _distance_matrix, _Rng)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- RNG -------------------------------------------------------------------
r = _Rng(1)
us = [r.uniform() for _ in range(1000)]
check("uniform in [0,1)", all(0.0 <= u < 1.0 for u in us))

# --- distance matrix and tour length ---------------------------------------
sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
D = _distance_matrix(sq)
check("distance matrix symmetric", all(D[i][j] == D[j][i] for i in range(4) for j in range(4)))
check("square side length", approx(D[0][1], 1.0, 1e-12))
# the perimeter tour 0-1-2-3 has length 4
check("tour_length of the square perimeter", approx(tour_length([0, 1, 2, 3], D), 4.0, 1e-12))
# a crossing tour is longer
check("crossing tour is longer", tour_length([0, 2, 1, 3], D) > 4.0)

# --- nearest-neighbour baseline is a valid tour ----------------------------
nn = nearest_neighbour_tour(D)
check("NN tour is a permutation", sorted(nn) == [0, 1, 2, 3])

# --- square: ACO finds the optimal perimeter -------------------------------
order, length = solve(sq, n_iter=30, seed=1)
check("square ACO finds perimeter 4", approx(length, 4.0, 1e-9))
check("square tour visits every city", sorted(order) == [0, 1, 2, 3])

# --- points on a circle: optimal is the regular polygon --------------------
m = 10
circ = [(math.cos(2 * math.pi * k / m), math.sin(2 * math.pi * k / m)) for k in range(m)]
order_c, length_c, hist = solve(circ, n_iter=60, seed=2, track=True)
opt = m * 2 * math.sin(math.pi / m)          # regular m-gon perimeter, unit circumradius
check("circle ACO finds the polygon perimeter", approx(length_c, opt, 1e-6))
check("circle tour is a permutation", sorted(order_c) == list(range(m)))

# --- best-so-far length is monotone non-increasing -------------------------
h = hist["history"]
check("best length monotone non-increasing", all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)))
check("history has one entry per iteration", len(h) == 60)

# --- ACO beats the nearest-neighbour greedy tour on random cities ----------
state = 9


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


cities = [(rng() * 10, rng() * 10) for _ in range(20)]
Dc = _distance_matrix(cities)
nn_len = tour_length(nearest_neighbour_tour(Dc), Dc)
_, aco_len = solve(cities, n_iter=100, seed=3)
check("ACO beats nearest-neighbour greedy", aco_len < nn_len)

# --- pheromone concentrates on short edges ---------------------------------
_, _, hist2 = solve(circ, n_iter=60, seed=2, track=True)
tau = hist2["pheromone"]
dist = hist2["dist"]
edges = sorted(((dist[i][j], i, j) for i in range(m) for j in range(i + 1, m)))
short_tau = sum(tau[i][j] for _, i, j in edges[:m]) / m
long_tau = sum(tau[i][j] for _, i, j in edges[-m:]) / m
check("pheromone is higher on short edges", short_tau > long_tau)

# --- pheromone matrix stays symmetric (symmetric TSP) ----------------------
check("pheromone symmetric", all(approx(tau[i][j], tau[j][i], 1e-12) for i in range(m) for j in range(m)))

# --- returned length matches the returned tour -----------------------------
best_order, best_len = solve(cities, n_iter=40, seed=5)
check("returned length matches tour", approx(tour_length(best_order, Dc), best_len, 1e-9))

# --- elitist vs non-elitist both produce valid tours -----------------------
_, l_elit = solve(cities, n_iter=60, elitist=True, seed=7)
_, l_plain = solve(cities, n_iter=60, elitist=False, seed=7)
check("elitist produces a valid short tour", l_elit < nn_len)
check("non-elitist also solves", l_plain < nn_len * 1.2)

# --- determinism -----------------------------------------------------------
check("ACO deterministic for a fixed seed",
      solve(cities, n_iter=30, seed=42) == solve(cities, n_iter=30, seed=42))

# --- tiny inputs -----------------------------------------------------------
check("single city", solve([(0.0, 0.0)], n_iter=5)[1] == 0.0)
check("two cities", approx(solve([(0.0, 0.0), (3.0, 0.0)], n_iter=5)[1], 6.0, 1e-9))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all ant_colony tests passed")
