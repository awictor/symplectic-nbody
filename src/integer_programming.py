"""Integer linear programming: branch and bound over the LP relaxation, where NP-hardness meets a solver.

Linear programming -- maximize c.x subject to A x <= b, x >= 0 -- is solved in polynomial time by the
simplex method. Add the innocent-looking demand that the variables be INTEGERS, and the problem becomes
NP-hard: it is exactly integer linear programming, the modeling language of scheduling, routing,
knapsacks, cutting stock, and half of operations research. There is no simplex for it, but there is a
beautiful exact method that turns the fast LP solver into an integer one: BRANCH AND BOUND.

The idea: solve the LP RELAXATION (drop the integrality). Its optimum is an upper bound on the integer
optimum (a larger feasible region can only help). If the relaxation happens to be all-integer, we are
done. Otherwise pick a fractional variable x_j = 3.4 and BRANCH into two subproblems, one with the
extra constraint x_j <= 3 and one with x_j >= 4 -- the fractional value 3.4 is excluded from both, but
every integer solution survives in one of them. Recurse. The BOUND part is the pruning that makes it
practical: keep the best integer solution found so far (the INCUMBENT); if a subproblem's LP
relaxation is already no better than the incumbent, discard the whole subtree unexplored. Good bounds
prune enormous parts of the search.

This module implements branch and bound on the repo's simplex LP relaxation, with best-bound-style
depth-first search, incumbent pruning, and optional per-variable bounds -- enough to solve small ILPs
and 0/1 problems (binary variables via 0 <= x <= 1 plus integrality) exactly. It is validated against
a brute-force integer search: it finds the true optimum on random small ILPs and 0/1 knapsacks; the
returned solution is integer and feasible; the LP relaxation bound always dominates the integer
optimum; adding integrality never improves the objective over the relaxation; and classic instances
(a knapsack, an assignment-style 0/1 problem) match hand-computed answers. Pure stdlib; the
integer-optimization companion to the simplex LP solver and the knapsack / branch-style tools."""

from __future__ import annotations

import math

from simplex import solve as lp_solve


_EPS = 1e-6


def _is_integer(v):
    return abs(v - round(v)) < 1e-5


def solve_ilp(c, constraints, maximize=True, integer_vars=None, var_bounds=None, max_nodes=100000):
    """Solve an integer linear program by branch and bound over the LP relaxation.

    c: objective coefficients. constraints: list of (a, sense, b) as in simplex.solve.
    integer_vars: indices required to be integer (default: all). var_bounds: dict j -> (lo, hi).
    Returns a dict with 'status', 'value', 'x', 'nodes'.
    """
    n = len(c)
    if integer_vars is None:
        integer_vars = set(range(n))
    else:
        integer_vars = set(integer_vars)

    # incumbent: best integer solution found so far
    best = {"value": None, "x": None}
    nodes = [0]

    def branch(extra):
        """extra: list of additional (a, sense, b) constraints bounding some variables."""
        if nodes[0] >= max_nodes:
            return
        nodes[0] += 1
        res = lp_solve(c, constraints + extra, maximize=maximize)
        if res.status != "optimal":
            return  # infeasible or unbounded subtree
        relax_val = res.value
        # bound: prune if the relaxation cannot beat the incumbent
        if best["value"] is not None:
            if maximize and relax_val <= best["value"] + _EPS:
                return
            if not maximize and relax_val >= best["value"] - _EPS:
                return
        x = res.x
        # find a fractional integer-constrained variable
        frac_j = None
        for j in integer_vars:
            if not _is_integer(x[j]):
                frac_j = j
                break
        if frac_j is None:
            # all-integer solution: candidate incumbent
            xr = [round(x[j]) if j in integer_vars else x[j] for j in range(n)]
            val = sum(c[j] * xr[j] for j in range(n))
            if best["value"] is None or (maximize and val > best["value"]) or \
               (not maximize and val < best["value"]):
                best["value"] = val
                best["x"] = xr
            return
        # branch on frac_j: floor and ceil
        f = x[frac_j]
        lo_row = [0.0] * n
        lo_row[frac_j] = 1.0
        # x_j <= floor(f)
        branch(extra + [(lo_row, "<=", math.floor(f))])
        # x_j >= ceil(f)
        branch(extra + [(lo_row, ">=", math.ceil(f))])

    # seed with any explicit variable bounds
    seed = []
    if var_bounds:
        for j, (lo, hi) in var_bounds.items():
            row = [0.0] * n
            row[j] = 1.0
            if lo is not None:
                seed.append((list(row), ">=", lo))
            if hi is not None:
                seed.append((list(row), "<=", hi))

    branch(seed)

    if best["value"] is None:
        return {"status": "infeasible", "value": None, "x": None, "nodes": nodes[0]}
    return {"status": "optimal", "value": best["value"], "x": best["x"], "nodes": nodes[0]}


def lp_relaxation_bound(c, constraints, maximize=True):
    """The LP-relaxation optimum -- an upper (maximize) / lower (minimize) bound on the ILP."""
    res = lp_solve(c, constraints, maximize=maximize)
    return res.value if res.status == "optimal" else None


def solve_knapsack_ilp(values, weights, capacity):
    """0/1 knapsack as an ILP: maximize sum v_i x_i s.t. sum w_i x_i <= capacity, x_i in {0,1}."""
    n = len(values)
    constraints = [(list(weights), "<=", capacity)]
    # x_i <= 1 for each item
    bounds = {i: (0, 1) for i in range(n)}
    return solve_ilp(values, constraints, maximize=True, integer_vars=set(range(n)),
                     var_bounds=bounds)


def brute_ilp(c, constraints, maximize, ranges):
    """Brute-force integer optimum over the given per-variable integer ranges (small instances)."""
    n = len(c)
    best_val = None
    best_x = None

    def feasible(x):
        for a, sense, b in constraints:
            lhs = sum(a[j] * x[j] for j in range(n))
            if sense == "<=" and lhs > b + 1e-9:
                return False
            if sense == ">=" and lhs < b - 1e-9:
                return False
            if sense == "=" and abs(lhs - b) > 1e-9:
                return False
        return True

    def rec(j, x):
        nonlocal best_val, best_x
        if j == n:
            if feasible(x) and all(v >= -1e-9 for v in x):
                val = sum(c[k] * x[k] for k in range(n))
                if best_val is None or (maximize and val > best_val) or \
                   (not maximize and val < best_val):
                    best_val = val
                    best_x = list(x)
            return
        lo, hi = ranges[j]
        for v in range(lo, hi + 1):
            x.append(v)
            rec(j + 1, x)
            x.pop()

    rec(0, [])
    return best_val, best_x
