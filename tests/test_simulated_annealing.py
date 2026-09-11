"""Tests for simulated_annealing: global optima, cooling, TSP, beats greedy, acceptance decay."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulated_annealing import (anneal, geometric_schedule, linear_schedule, solve_tsp,
                                 tour_length, nearest_neighbour_tour, _reverse_segment, _Rng)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- cooling schedules -----------------------------------------------------
g = geometric_schedule(10.0, 0.9)
check("geometric start", approx(g(0), 10.0, 1e-9))
check("geometric decays", g(0) > g(5) > g(20) > 0)
check("geometric ratio", approx(g(1) / g(0), 0.9, 1e-9))
lin = linear_schedule(10.0, 100)
check("linear start", approx(lin(0), 10.0, 1e-9))
check("linear reaches ~0", lin(100) < 1e-6)
check("linear monotone", lin(10) > lin(50) > lin(90))

# --- RNG sanity ------------------------------------------------------------
r = _Rng(3)
us = [r.uniform() for _ in range(1000)]
check("uniform in [0,1)", all(0.0 <= u < 1.0 for u in us))
check("uniform roughly centered", abs(sum(us) / len(us) - 0.5) < 0.05)
check("randint in range", all(0 <= _Rng(i).randint(7) < 7 for i in range(20)))

# --- global minimum of a multimodal function ------------------------------
def f(x):
    return x[0] ** 2 * 0.05 - 3 * math.cos(x[0]) + 3 * math.sin(1.3 * x[0])


def neigh(x, rng):
    return [x[0] + rng.gauss(1.5)]


# brute-force reference
grid = [-20 + i * 0.01 for i in range(4000)]
gx = min(grid, key=lambda x: f([x]))
gmin = f([gx])

best, cost = anneal([15.0], f, neigh, geometric_schedule(10.0, 0.999), 8000, seed=1)
check("SA reaches near the global minimum", cost <= gmin + 0.1)
check("SA locates the global minimizer", approx(best[0], gx, 0.3))

# --- SA escapes where greedy descent gets stuck ----------------------------
def greedy(x0):
    x = x0
    for _ in range(2000):
        nxt = x
        for dx in (-0.05, 0.05):
            if f([x + dx]) < f([nxt]):
                nxt = x + dx
        if nxt == x:
            break
        x = nxt
    return x


gstuck = greedy(15.0)
check("greedy gets stuck in a local min", f([gstuck]) > gmin + 0.5)
check("SA beats greedy from the same start", cost < f([gstuck]))

# --- travelling salesman: a unit square has optimal perimeter 4 ------------
square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
order, length = solve_tsp(square, steps=6000, seed=2)
check("square TSP finds perimeter 4", approx(length, 4.0, 1e-6))
check("TSP tour visits every city once", sorted(order) == list(range(4)))

# --- a colinear set: optimal tour is out-and-back = 2*(max-min) ------------
line = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
_, line_len = solve_tsp(line, steps=6000, seed=5)
check("colinear TSP is out-and-back (length 8)", approx(line_len, 8.0, 1e-6))

# --- SA beats the nearest-neighbour greedy tour on random cities -----------
rng = _Rng(9)
cities = [(rng.uniform() * 10, rng.uniform() * 10) for _ in range(20)]
nn = nearest_neighbour_tour(cities)
nn_len = tour_length(nn, cities)
sa_order, sa_len = solve_tsp(cities, steps=30000, seed=3)
check("SA tour beats nearest-neighbour greedy", sa_len < nn_len)
check("SA tour is a valid permutation", sorted(sa_order) == list(range(20)))

# --- segment reversal preserves the permutation ----------------------------
perm = list(range(10))
rv = _reverse_segment(perm, _Rng(1))
check("reversal keeps a permutation", sorted(rv) == list(range(10)))
check("reversal same length", len(rv) == 10)

# --- acceptance rate falls as the system cools -----------------------------
_, _, hist = anneal([15.0], f, neigh, geometric_schedule(10.0, 0.999), 4000, seed=1, track=True)
early = sum(hist["accept_flags"][:500]) / 500
late = sum(hist["accept_flags"][-500:]) / 500
check("acceptance rate decays as it cools", late < early)
check("early acceptance is high (exploration)", early > 0.5)
check("overall accept rate reported", 0.0 <= hist["accept_rate"] <= 1.0)

# --- best cost never exceeds the initial cost ------------------------------
b2, c2 = anneal([15.0], f, neigh, geometric_schedule(5.0, 0.99), 3000, seed=7)
check("SA never worsens the starting cost", c2 <= f([15.0]) + 1e-9)

# --- determinism -----------------------------------------------------------
r1 = anneal([15.0], f, neigh, geometric_schedule(10.0, 0.999), 2000, seed=42)
r2 = anneal([15.0], f, neigh, geometric_schedule(10.0, 0.999), 2000, seed=42)
check("simulated annealing deterministic for a fixed seed", r1 == r2)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all simulated_annealing tests passed")
