"""Tests for genetic_algorithm: OneMax, real optimization, knapsack, elitism, beats random."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from genetic_algorithm import (genetic_algorithm, binary_init, binary_mutate, real_init,
                               real_mutate, solve_knapsack, _tournament, _one_point_crossover,
                               _Rng)

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
check("randint in range", all(0 <= _Rng(i).randint(5) < 5 for i in range(20)))

# --- crossover preserves genes ---------------------------------------------
a = [0, 0, 0, 0, 0]
b = [1, 1, 1, 1, 1]
c1, c2 = _one_point_crossover(a, b, _Rng(3))
check("crossover children have full length", len(c1) == 5 and len(c2) == 5)
# every gene came from one parent; the two children are complementary at each position
check("crossover is complementary", all(c1[i] + c2[i] == 1 for i in range(5)))

# --- tournament selection favours fitness ----------------------------------
pop = [[0], [1], [2], [3]]
fits = [0, 1, 2, 3]
picks = [_tournament(pop, fits, 3, _Rng(i))[0] for i in range(200)]
check("tournament prefers high fitness", sum(picks) / len(picks) > 2.0)

# --- OneMax: maximize the number of 1 bits ---------------------------------
L = 40
best, fit = genetic_algorithm(lambda bits: sum(bits), binary_init(L), binary_mutate(1.0 / L),
                              n_generations=120, pop_size=60, seed=1)
check("OneMax reaches all ones", fit == L)
check("OneMax solution is all ones", all(g == 1 for g in best))

# --- real-valued multimodal maximization -----------------------------------
def f(x):
    # negative of a bumpy function -> maximizing finds its global minimum
    return -(x[0] ** 2 * 0.05 - 3 * math.cos(x[0]) + 3 * math.sin(1.3 * x[0]))


bounds = [(-20.0, 20.0)]
best_r, fit_r, hist = genetic_algorithm(f, real_init(bounds), real_mutate(1.0, bounds),
                                        n_generations=80, pop_size=50, seed=2, track=True)
grid = [-20 + i * 0.01 for i in range(4000)]
gx = max(grid, key=lambda x: f([x]))
check("real GA reaches near the global optimum", fit_r >= f([gx]) - 0.1)
check("real GA locates the optimizer", approx(best_r[0], gx, 0.3))
check("real GA stays within bounds", -20.0 <= best_r[0] <= 20.0)

# --- elitism makes the best fitness monotone non-decreasing ----------------
bh = hist["best_history"]
check("best fitness monotone under elitism",
      all(bh[i + 1] >= bh[i] - 1e-9 for i in range(len(bh) - 1)))
check("population mean fitness improves", hist["mean_history"][-1] > hist["mean_history"][0])

# --- knapsack matches the brute-force optimum ------------------------------
w = [2, 3, 4, 5, 9]
v = [3, 4, 5, 8, 10]
cap = 10
brute = 0
for m in range(2 ** len(w)):
    bits = [(m >> i) & 1 for i in range(len(w))]
    tw = sum(w[i] for i in range(len(w)) if bits[i])
    if tw <= cap:
        brute = max(brute, sum(v[i] for i in range(len(w)) if bits[i]))
gbits, gv, gw = solve_knapsack(w, v, cap, seed=3)
check("knapsack GA matches brute-force optimum", gv == brute)
check("knapsack solution respects capacity", gw <= cap)

# --- larger knapsack stays feasible and good -------------------------------
state = 5


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


wb = [int(rng() * 20) + 1 for _ in range(15)]
vb = [int(rng() * 20) + 1 for _ in range(15)]
capb = sum(wb) // 2
_, vv, ww = solve_knapsack(wb, vb, capb, n_generations=300, pop_size=100, seed=8)
check("large knapsack stays within capacity", ww <= capb)
check("large knapsack collects positive value", vv > 0)

# --- GA beats random search at equal budget (OneMax) -----------------------
budget = 60 * 60      # pop_size * generations
rr = _Rng(7)
rand_best = 0
for _ in range(budget):
    rand_best = max(rand_best, sum(rr.randint(2) for _ in range(L)))
ga_best, _ = genetic_algorithm(lambda bits: sum(bits), binary_init(L), binary_mutate(1.0 / L),
                               n_generations=60, pop_size=60, seed=9)
check("GA beats random search at equal budget", ga_best.count(1) > rand_best)

# --- determinism -----------------------------------------------------------
r1 = genetic_algorithm(lambda b: sum(b), binary_init(20), binary_mutate(0.05),
                       n_generations=30, pop_size=30, seed=42)
r2 = genetic_algorithm(lambda b: sum(b), binary_init(20), binary_mutate(0.05),
                       n_generations=30, pop_size=30, seed=42)
check("genetic algorithm deterministic for a fixed seed", r1 == r2)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all genetic_algorithm tests passed")
