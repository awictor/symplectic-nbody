"""Tests for bayes_opt: expected improvement, convergence to known minima, beats random search."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bayes_opt import (minimize, random_search, expected_improvement,
                       _norm_cdf, _norm_pdf)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- normal CDF / PDF ------------------------------------------------------
check("normal cdf at 0 is 0.5", approx(_norm_cdf(0.0), 0.5, 1e-9))
check("normal cdf is monotone", _norm_cdf(-1.0) < _norm_cdf(0.0) < _norm_cdf(1.0))
check("normal cdf tails", _norm_cdf(5.0) > 0.999 and _norm_cdf(-5.0) < 0.001)
check("normal pdf at 0", approx(_norm_pdf(0.0), 1.0 / math.sqrt(2 * math.pi), 1e-9))
check("normal pdf symmetric", approx(_norm_pdf(1.3), _norm_pdf(-1.3), 1e-12))

# --- expected improvement --------------------------------------------------
check("EI zero at zero uncertainty", expected_improvement(1.0, 0.0, 2.0) == 0.0)
check("EI positive for a promising uncertain point", expected_improvement(0.5, 0.5, 2.0) > 0)
# a point predicted worse than best, but uncertain, still has some EI (exploration)
check("EI rewards exploration", expected_improvement(3.0, 1.0, 2.0) > 0)
# more uncertainty at the same mean => at least as much EI
check("EI increases with uncertainty",
      expected_improvement(2.0, 1.0, 2.0) >= expected_improvement(2.0, 0.5, 2.0))
# a confident point right at the current best offers ~no improvement
check("EI tiny for confident best-matching point",
      expected_improvement(2.0, 1e-6, 2.0) < 1e-3)

# --- 1-D multimodal: f(x) = sin(x) + sin(10x/3) on [2.7, 7.5] ---------------
def f1(v):
    return math.sin(v[0]) + math.sin(10 * v[0] / 3)


# brute-force reference minimum
grid = [2.7 + i * 0.001 for i in range(4801)]
x_true = min(grid, key=lambda x: f1([x]))
y_true = f1([x_true])

res = minimize(f1, [(2.7, 7.5)], n_init=4, n_iter=16, seed=1)
check("1-D BO finds near-optimal value", res["best_y"] <= y_true + 0.05)
check("1-D BO locates the minimizer", approx(res["best_x"][0], x_true, 0.1))
check("1-D BO used exactly n_init + n_iter evals", res["n_eval"] == 20)
check("history length matches n_eval", len(res["X"]) == res["n_eval"] == len(res["y"]))

# --- 2-D Branin, known global minimum value ~0.397887 ----------------------
def branin(v):
    x, y = v
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    t = 1 / (8 * math.pi)
    return (y - b * x * x + c * x - 6) ** 2 + 10 * (1 - t) * math.cos(x) + 10


BRANIN_MIN = 0.397887
resb = minimize(branin, [(-5.0, 10.0), (0.0, 15.0)], n_init=8, n_iter=24, seed=2)
check("2-D Branin BO gets close to global min", resb["best_y"] <= BRANIN_MIN + 0.3)
check("Branin best is inside the domain",
      -5.0 <= resb["best_x"][0] <= 10.0 and 0.0 <= resb["best_x"][1] <= 15.0)

# --- BO beats random search on average given the same budget ---------------
bo_vals, rs_vals = [], []
for s in range(5):
    rb = minimize(branin, [(-5.0, 10.0), (0.0, 15.0)], n_init=8, n_iter=22, seed=s)
    rs = random_search(branin, [(-5.0, 10.0), (0.0, 15.0)], n_eval=rb["n_eval"], seed=s + 100)
    bo_vals.append(rb["best_y"])
    rs_vals.append(rs["best_y"])
mean_bo = sum(bo_vals) / len(bo_vals)
mean_rs = sum(rs_vals) / len(rs_vals)
check("BO mean beats random-search mean", mean_bo < mean_rs)
check("BO is reliably near the optimum (all runs)", all(v <= BRANIN_MIN + 0.5 for v in bo_vals))
# median comparison is robust to random search's occasional lucky run
bo_sorted = sorted(bo_vals)
rs_sorted = sorted(rs_vals)
check("BO median beats random-search median", bo_sorted[2] < rs_sorted[2])

# --- returns the actual best seen, and it is consistent with history -------
check("best_y equals min of history", approx(resb["best_y"], min(resb["y"]), 1e-12))
i = resb["y"].index(min(resb["y"]))
check("best_x corresponds to best_y", resb["best_x"] == resb["X"][i])

# --- a trivial convex bowl is solved almost exactly ------------------------
bowl = minimize(lambda v: (v[0] - 3.0) ** 2, [(-10.0, 10.0)], n_init=4, n_iter=16, seed=7)
check("convex bowl minimized near x=3", approx(bowl["best_x"][0], 3.0, 0.3))
check("convex bowl value near 0", bowl["best_y"] < 0.1)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bayes_opt tests passed")
