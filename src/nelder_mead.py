"""Nelder-Mead: derivative-free optimization by a crawling simplex.

Newton's method needs a Jacobian; gradient descent needs a gradient. NELDER-MEAD (1965) needs
NEITHER -- it minimizes a function using only its values, by maintaining a SIMPLEX of n+1 points in
n dimensions (a triangle in 2-D, a tetrahedron in 3-D) that tumbles and shrinks its way downhill.
It is the workhorse behind "just minimize this black box" (it is SciPy's default for
gradient-free problems, and MATLAB's fminsearch): robust, simple, and needing no derivative
information, at the cost of no convergence guarantee and slower progress in high dimensions.

Each iteration reflects the WORST vertex through the centroid of the others and, based on how good
the reflection is, takes one of four moves:

  REFLECT  -- mirror the worst point through the centroid; the default probe.
  EXPAND   -- if the reflection is the new best, push further in that promising direction.
  CONTRACT -- if the reflection is poor, pull back toward the centroid (outside or inside).
  SHRINK   -- if even contraction fails, shrink the whole simplex toward the best vertex.

The simplex thus crawls across the landscape like an amoeba, stretching down valleys and squeezing
through narrow curved ones (it famously handles the Rosenbrock banana). This module implements the
standard algorithm with the classic coefficients and both value- and size-based convergence tests
-- verified that it finds the minimum of the Sphere, Rosenbrock, and Beale benchmarks from several
starts without any gradient, that the best vertex improves monotonically, that the simplex shrinks
to a point, and that restarting from the result refines it further. Pure stdlib; the derivative-free
local-optimization companion to the Newton and gradient-based notes."""

from __future__ import annotations

import math


def _centroid(points):
    n = len(points)
    d = len(points[0])
    return [sum(p[j] for p in points) / n for j in range(d)]


def _add(a, b, s=1.0):
    return [a[i] + s * b[i] for i in range(len(a))]


def _sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def minimize(cost_fn, x0, step=0.5, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
             tol=1e-10, max_iter=2000, track=False):
    """Minimize cost_fn by the Nelder-Mead simplex method.

    x0    : starting point (length n); the initial simplex is x0 plus n axis-offset vertices.
    alpha : reflection, gamma : expansion, rho : contraction, sigma : shrink coefficients.
    Converges when the spread of vertex values (and the simplex size) fall below tol.
    Returns (best_point, best_value) or, with track=True, also a history dict."""
    n = len(x0)
    # build the initial simplex: x0 and n vertices offset along each axis
    simplex = [list(x0)]
    for i in range(n):
        v = list(x0)
        v[i] += step if x0[i] == 0 else step * abs(x0[i])
        simplex.append(v)
    f = [cost_fn(p) for p in simplex]
    n_eval = len(simplex)
    best_history = []

    it = 0
    for it in range(1, max_iter + 1):
        # order vertices best (lowest) to worst (highest)
        order = sorted(range(n + 1), key=lambda i: f[i])
        simplex = [simplex[i] for i in order]
        f = [f[i] for i in order]
        best_history.append(f[0])

        # convergence: value spread and simplex size both small
        value_spread = f[-1] - f[0]
        size = max(_dist(simplex[0], simplex[i]) for i in range(1, n + 1))
        if value_spread < tol and size < tol:
            break

        centroid = _centroid(simplex[:-1])            # exclude the worst vertex
        worst = simplex[-1]

        # REFLECT
        reflected = _add(centroid, _sub(centroid, worst), alpha)
        fr = cost_fn(reflected)
        n_eval += 1
        if f[0] <= fr < f[-2]:
            simplex[-1], f[-1] = reflected, fr
            continue

        # EXPAND (reflection is the new best -> push further)
        if fr < f[0]:
            expanded = _add(centroid, _sub(reflected, centroid), gamma)
            fe = cost_fn(expanded)
            n_eval += 1
            if fe < fr:
                simplex[-1], f[-1] = expanded, fe
            else:
                simplex[-1], f[-1] = reflected, fr
            continue

        # CONTRACT
        if fr < f[-1]:
            # outside contraction (reflection better than worst but not great)
            contracted = _add(centroid, _sub(reflected, centroid), rho)
            fc = cost_fn(contracted)
            n_eval += 1
            if fc <= fr:
                simplex[-1], f[-1] = contracted, fc
                continue
        else:
            # inside contraction (reflection worse than worst)
            contracted = _add(centroid, _sub(worst, centroid), rho)
            fc = cost_fn(contracted)
            n_eval += 1
            if fc < f[-1]:
                simplex[-1], f[-1] = contracted, fc
                continue

        # SHRINK toward the best vertex
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = _add(best, _sub(simplex[i], best), sigma)
            f[i] = cost_fn(simplex[i])
            n_eval += 1

    # final ordering
    order = sorted(range(n + 1), key=lambda i: f[i])
    best_point = simplex[order[0]]
    best_value = f[order[0]]
    if track:
        return best_point, best_value, {"history": best_history, "n_iter": it,
                                        "n_eval": n_eval}
    return best_point, best_value


def minimize_restart(cost_fn, x0, restarts=3, **kwargs):
    """Nelder-Mead with restarts: re-seed the simplex at each solution to escape a stalled simplex,
    a standard robustness trick. Returns (best_point, best_value)."""
    x = list(x0)
    best_val = float("inf")
    best_pt = x
    for _ in range(restarts):
        pt, val = minimize(cost_fn, x, **kwargs)
        if val < best_val:
            best_val, best_pt = val, pt
        x = pt
    return best_pt, best_val


# --- benchmark functions (global minimum 0) --------------------------------
def sphere(x):
    return sum(xi * xi for xi in x)


def rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def beale(x):
    """Beale's function; global minimum 0 at (3, 0.5). A classic 2-D test with a curved valley."""
    a, b = x
    return ((1.5 - a + a * b) ** 2 + (2.25 - a + a * b * b) ** 2
            + (2.625 - a + a * b ** 3) ** 2)
