"""Tests for tsp: Held-Karp vs brute force, 2-opt near-optimality, valid tours."""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tsp import (euclidean_matrix, held_karp, nearest_neighbour, two_opt, solve_2opt,
                 tour_length)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 31337
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_tsp(D):
    n = len(D)
    best = math.inf
    best_tour = None
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm)
        L = tour_length(tour, D)
        if L < best:
            best = L
            best_tour = tour
    return best_tour, best


# --- Held-Karp matches brute force -----------------------------------------
ok = True
for _ in range(20):
    n = 4 + int(rng() * 5)         # 4..8 cities
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    D = euclidean_matrix(pts)
    _, hk_len = held_karp(D)
    _, bf_len = brute_tsp(D)
    if abs(hk_len - bf_len) > 1e-6:
        ok = False
        break
check("Held-Karp matches brute force over 20 random instances", ok)

# --- Held-Karp returns a valid tour ----------------------------------------
pts = [(rng() * 100, rng() * 100) for _ in range(10)]
D = euclidean_matrix(pts)
tour, length = held_karp(D)
check("Held-Karp tour is a valid permutation", sorted(tour) == list(range(10)))
check("Held-Karp tour starts at 0", tour[0] == 0)
check("reported length matches summed edges", abs(tour_length(tour, D) - length) < 1e-9)

# --- 2-opt improves the nearest-neighbour tour -----------------------------
improved_count = 0
for _ in range(20):
    n = 10 + int(rng() * 15)
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    D = euclidean_matrix(pts)
    nn = nearest_neighbour(D)
    nn_len = tour_length(nn, D)
    opt2 = two_opt(nn, D)
    opt2_len = tour_length(opt2, D)
    if opt2_len <= nn_len + 1e-9:
        improved_count += 1
    # 2-opt should never make it worse
    check_worse = opt2_len <= nn_len + 1e-9
    if not check_worse:
        break
check(f"2-opt never worsens the tour (improved on {improved_count}/20)", improved_count == 20)

# --- 2-opt gets within a small ratio of the Held-Karp optimum --------------
ratios = []
for _ in range(15):
    n = 8 + int(rng() * 4)         # small enough for Held-Karp
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    D = euclidean_matrix(pts)
    _, hk = held_karp(D)
    _, l2 = solve_2opt(D)
    ratios.append(l2 / hk)
avg_ratio = sum(ratios) / len(ratios)
check(f"2-opt averages within ~10% of optimal (avg ratio {avg_ratio:.3f})", avg_ratio < 1.10)
check("2-opt is never below optimal", all(r >= 1 - 1e-9 for r in ratios))

# --- 2-opt tours are valid permutations ------------------------------------
ok = True
for _ in range(20):
    n = 5 + int(rng() * 20)
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    D = euclidean_matrix(pts)
    tour, _ = solve_2opt(D)
    if sorted(tour) != list(range(n)):
        ok = False
        break
check("2-opt tours are valid permutations", ok)

# --- Euclidean matrix satisfies the triangle inequality --------------------
pts = [(rng() * 100, rng() * 100) for _ in range(15)]
D = euclidean_matrix(pts)
n = len(D)
tri_ok = all(D[i][k] <= D[i][j] + D[j][k] + 1e-9
             for i in range(n) for j in range(n) for k in range(n))
check("Euclidean distance matrix satisfies the triangle inequality", tri_ok)
check("distance matrix is symmetric", all(abs(D[i][j] - D[j][i]) < 1e-12 for i in range(n) for j in range(n)))
check("diagonal is zero", all(D[i][i] == 0 for i in range(n)))

# --- known tiny instances --------------------------------------------------
# a unit square: optimal tour is the perimeter, length 4
square = euclidean_matrix([(0, 0), (1, 0), (1, 1), (0, 1)])
_, sq_len = held_karp(square)
check("unit square optimal tour has length 4", abs(sq_len - 4.0) < 1e-9)

# a collinear set: optimal tour goes out and back
line = euclidean_matrix([(0, 0), (1, 0), (2, 0), (3, 0)])
_, line_len = held_karp(line)
check("collinear points: optimal tour is out-and-back (length 6)", abs(line_len - 6.0) < 1e-9)

# --- 2-opt matches Held-Karp exactly on the square -------------------------
_, sq2 = solve_2opt(square)
check("2-opt finds the optimal square tour", abs(sq2 - 4.0) < 1e-9)

# --- single and two cities -------------------------------------------------
check("single city: zero-length tour", held_karp([[0.0]])[1] == 0.0)
check("two cities: round trip", abs(held_karp([[0, 5], [5, 0]])[1] - 10.0) < 1e-9)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all tsp tests passed")
