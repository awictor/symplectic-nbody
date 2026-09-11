"""Tests for differential_evolution: benchmark optima, monotone best, beats random, bounds, dim."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from differential_evolution import (minimize, random_search, sphere, rastrigin,
                                    rosenbrock, _reflect, _Rng)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- RNG + helpers ---------------------------------------------------------
r = _Rng(1)
check("uniform in range", all(-2.0 <= _Rng(i).uniform(-2.0, 3.0) < 3.0 for i in range(20)))
idx = r.sample3(10, 4)
check("sample3 returns 3 distinct indices", len(set(idx)) == 3)
check("sample3 excludes the target", 4 not in idx)

# --- bound reflection ------------------------------------------------------
check("reflect below low", approx(_reflect(-1.0, 0.0, 5.0), 1.0, 1e-12))
check("reflect above high", approx(_reflect(6.0, 0.0, 5.0), 4.0, 1e-12))
check("reflect in-bounds is identity", _reflect(2.5, 0.0, 5.0) == 2.5)
check("reflect lands inside", all(0.0 <= _reflect(v, 0.0, 5.0) <= 5.0 for v in (-3.2, 7.9, 12.1)))

# --- Sphere: smooth bowl ---------------------------------------------------
best, cost = minimize(sphere, [(-5.0, 5.0), (-5.0, 5.0)], n_iter=200, seed=1)
check("DE solves Sphere to ~0", cost < 1e-4)
check("DE Sphere near the origin", all(abs(x) < 0.05 for x in best))

# --- Rastrigin: highly multimodal ------------------------------------------
best_r, cost_r, hist = minimize(rastrigin, [(-5.12, 5.12)] * 2, n_iter=300, seed=2, track=True)
check("DE solves multimodal Rastrigin", cost_r < 0.5)

# --- best cost is monotone non-increasing (greedy selection) ---------------
h = hist["history"]
check("best cost monotone non-increasing", all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)))
check("cost falls substantially", h[-1] < 0.1 * h[0] + 1e-9)
check("history length is n_iter + 1", len(h) == 301)

# --- Rosenbrock: curved valley, minimum at (1,1) ---------------------------
best_ro, cost_ro = minimize(rosenbrock, [(-2.0, 2.0), (-1.0, 3.0)], n_iter=400, seed=3)
check("DE solves Rosenbrock", cost_ro < 0.01)
check("DE Rosenbrock finds (1,1)", approx(best_ro[0], 1.0, 0.1) and approx(best_ro[1], 1.0, 0.1))

# --- higher dimensions -----------------------------------------------------
_, cost5 = minimize(sphere, [(-5.0, 5.0)] * 5, n_iter=300, seed=7)
check("DE solves 5-D Sphere", cost5 < 1e-3)
_, cost10 = minimize(sphere, [(-5.0, 5.0)] * 10, pop_size=50, n_iter=800, seed=1)
check("DE solves 10-D Sphere", cost10 < 0.01)

# --- DE beats random search at equal budget --------------------------------
for fn, bd, name in [(sphere, [(-5.0, 5.0)] * 2, "Sphere"),
                     (rastrigin, [(-5.12, 5.12)] * 2, "Rastrigin")]:
    _, de = minimize(fn, bd, pop_size=30, n_iter=200, seed=5)
    _, rs = random_search(fn, bd, n_eval=30 * 200, seed=6)
    check(f"DE beats random search on {name}", de < rs)

# --- solution stays inside the bounds --------------------------------------
best_b, _ = minimize(lambda x: -(x[0] + x[1]), [(-3.0, 3.0), (-3.0, 3.0)],
                     pop_size=20, n_iter=100, seed=8)
check("solution respects the box bounds", all(-3.0 <= v <= 3.0 for v in best_b))
check("maximizing pull reaches the corner",
      approx(best_b[0], 3.0, 0.05) and approx(best_b[1], 3.0, 0.05))

# --- returned cost matches the returned vector -----------------------------
b9, c9 = minimize(sphere, [(-5.0, 5.0)] * 3, pop_size=25, n_iter=150, seed=9)
check("returned cost matches vector", approx(sphere(b9), c9, 1e-9))

# --- default population scales with dimension ------------------------------
# (indirectly: a 4-D problem still solves with the default pop_size)
_, c4 = minimize(sphere, [(-5.0, 5.0)] * 4, n_iter=300, seed=11)
check("default population solves 4-D", c4 < 1e-2)

# --- determinism -----------------------------------------------------------
r1 = minimize(sphere, [(-5.0, 5.0)] * 2, pop_size=20, n_iter=100, seed=42)
r2 = minimize(sphere, [(-5.0, 5.0)] * 2, pop_size=20, n_iter=100, seed=42)
check("DE deterministic for a fixed seed", r1 == r2)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all differential_evolution tests passed")
