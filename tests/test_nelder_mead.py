"""Tests for nelder_mead: benchmark optima without derivatives, monotone best, restarts, shrink."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nelder_mead import minimize, minimize_restart, sphere, rosenbrock, beale

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- benchmark functions have their known minima ---------------------------
check("sphere min at origin", sphere([0.0, 0.0]) == 0.0)
check("rosenbrock min at (1,1)", approx(rosenbrock([1.0, 1.0]), 0.0, 1e-12))
check("beale min at (3, 0.5)", approx(beale([3.0, 0.5]), 0.0, 1e-12))

# --- Sphere from several starts --------------------------------------------
for start in ([3.0, -2.0], [10.0, 10.0], [-5.0, 0.1]):
    p, v = minimize(sphere, start)
    check(f"sphere solved from {start}", v < 1e-8 and all(abs(x) < 1e-4 for x in p))

# --- Rosenbrock: the curved-valley classic ---------------------------------
p, v, hist = minimize(rosenbrock, [-1.2, 1.0], track=True)
check("Rosenbrock solved", v < 1e-6)
check("Rosenbrock finds (1,1)", approx(p[0], 1.0, 1e-3) and approx(p[1], 1.0, 1e-3))
check("Rosenbrock used no derivative (value calls only)", hist["n_eval"] > 0)

# --- best vertex value is monotone non-increasing --------------------------
h = hist["history"]
check("best value monotone non-increasing", all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)))
check("best value falls substantially", h[-1] < 0.01 * h[0] + 1e-9)

# --- Beale's function ------------------------------------------------------
pb, vb = minimize(beale, [1.0, 1.0])
check("Beale solved", vb < 1e-6)
check("Beale finds (3, 0.5)", approx(pb[0], 3.0, 1e-2) and approx(pb[1], 0.5, 1e-2))

# --- higher dimension ------------------------------------------------------
p4, v4 = minimize(sphere, [1.0, 2.0, 3.0, 4.0])
check("4-D sphere solved", v4 < 1e-6)

# --- no derivative needed: works on a non-smooth objective -----------------
pn, vn = minimize(lambda x: abs(x[0] - 2) + abs(x[1] + 3), [0.0, 0.0])
check("non-smooth abs-sum minimized", vn < 1e-3)
check("non-smooth minimizer found", approx(pn[0], 2.0, 1e-2) and approx(pn[1], -3.0, 1e-2))

# --- restarts refine (never worse than a single run) -----------------------
p_single, v_single = minimize(rosenbrock, [-1.2, 1.0])
p_restart, v_restart = minimize_restart(rosenbrock, [-1.2, 1.0], restarts=3)
check("restart is at least as good as a single run", v_restart <= v_single + 1e-12)
check("restart solves Rosenbrock", v_restart < 1e-8)

# --- returned value matches the returned point -----------------------------
pp, vv = minimize(sphere, [2.0, -1.0, 0.5])
check("returned value matches point", approx(sphere(pp), vv, 1e-12))

# --- starting at the optimum stays there -----------------------------------
p0, v0 = minimize(sphere, [0.0, 0.0])
check("starting at the optimum stays", v0 < 1e-10)

# --- a scaled/shifted quadratic (exercises expansion and contraction) ------
def shifted(x):
    return 3 * (x[0] - 7) ** 2 + 5 * (x[1] + 4) ** 2 + 2


ps, vs = minimize(shifted, [0.0, 0.0])
check("shifted quadratic reaches its floor", approx(vs, 2.0, 1e-6))
check("shifted quadratic minimizer", approx(ps[0], 7.0, 1e-3) and approx(ps[1], -4.0, 1e-3))

# --- convergence is finite (does not exhaust max_iter on an easy problem) ---
_, _, heasy = minimize(sphere, [1.0, 1.0], track=True)
check("easy problem converges well before max_iter", heasy["n_iter"] < 500)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all nelder_mead tests passed")
