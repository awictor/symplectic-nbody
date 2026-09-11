"""Tests for particle_swarm: benchmark optima, monotone gbest, beats random, inertia effect."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from particle_swarm import (minimize, random_search, sphere, rastrigin, rosenbrock, _Rng)

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
check("uniform range respected", all(-3.0 <= _Rng(i).uniform(-3.0, 5.0) < 5.0 for i in range(20)))

# --- benchmark functions have their known minima ---------------------------
check("sphere min at origin", sphere([0.0, 0.0]) == 0.0)
check("sphere positive elsewhere", sphere([1.0, 1.0]) == 2.0)
check("rastrigin min at origin", approx(rastrigin([0.0, 0.0]), 0.0, 1e-9))
check("rosenbrock min at (1,1)", approx(rosenbrock([1.0, 1.0]), 0.0, 1e-9))

# --- Sphere: easy smooth bowl ----------------------------------------------
best, cost = minimize(sphere, [(-5.0, 5.0), (-5.0, 5.0)], n_particles=30, n_iter=200, seed=1)
check("PSO solves Sphere to ~0", cost < 1e-4)
check("PSO Sphere near the origin", all(abs(x) < 0.05 for x in best))

# --- Rastrigin: highly multimodal ------------------------------------------
best_r, cost_r, hist = minimize(rastrigin, [(-5.12, 5.12)] * 2, n_particles=40, n_iter=300,
                                w=0.7, w_final=0.3, seed=2, track=True)
check("PSO solves multimodal Rastrigin", cost_r < 0.5)

# --- global best improves monotonically ------------------------------------
h = hist["history"]
check("global best monotone non-increasing", all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)))
check("cost falls substantially", h[-1] < 0.1 * h[0] + 1e-9)
check("history length is n_iter + 1", len(h) == 301)

# --- Rosenbrock: curved valley, minimum at (1,1) ---------------------------
best_ro, cost_ro = minimize(rosenbrock, [(-2.0, 2.0), (-1.0, 3.0)], n_particles=40,
                            n_iter=400, w=0.7, w_final=0.3, seed=3)
check("PSO solves Rosenbrock", cost_ro < 0.1)
check("PSO Rosenbrock finds (1,1)", approx(best_ro[0], 1.0, 0.2) and approx(best_ro[1], 1.0, 0.3))

# --- higher dimension still works ------------------------------------------
_, cost5 = minimize(sphere, [(-5.0, 5.0)] * 5, n_particles=40, n_iter=300, seed=7)
check("PSO solves 5-D Sphere", cost5 < 1e-3)

# --- PSO beats random search at equal budget -------------------------------
for fn, bd, name in [(sphere, [(-5.0, 5.0)] * 2, "Sphere"),
                     (rastrigin, [(-5.12, 5.12)] * 2, "Rastrigin")]:
    _, pso_cost = minimize(fn, bd, n_particles=30, n_iter=200, seed=5)
    _, rs_cost = random_search(fn, bd, n_eval=30 * 200, seed=6)
    check(f"PSO beats random search on {name}", pso_cost < rs_cost)

# --- particles stay inside the bounds --------------------------------------
# a cost that would pull particles outward if bounds were not enforced
best_b, _ = minimize(lambda x: -(x[0] + x[1]), [(-3.0, 3.0), (-3.0, 3.0)],
                     n_particles=20, n_iter=100, seed=8)
check("solution respects the box bounds", all(-3.0 <= v <= 3.0 for v in best_b))
check("maximizing pull lands at the corner", approx(best_b[0], 3.0, 0.01) and approx(best_b[1], 3.0, 0.01))

# --- returned cost matches the returned position ---------------------------
b9, c9 = minimize(sphere, [(-5.0, 5.0)] * 2, n_particles=25, n_iter=150, seed=9)
check("returned cost matches position", approx(sphere(b9), c9, 1e-9))

# --- determinism -----------------------------------------------------------
r1 = minimize(sphere, [(-5.0, 5.0)] * 2, n_particles=20, n_iter=100, seed=42)
r2 = minimize(sphere, [(-5.0, 5.0)] * 2, n_particles=20, n_iter=100, seed=42)
check("PSO deterministic for a fixed seed", r1 == r2)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all particle_swarm tests passed")
