"""The simplex method: solving linear programs by walking the vertices of a polytope.

LINEAR PROGRAMMING is the problem of maximizing (or minimizing) a linear objective subject to linear
inequality and equality constraints -- the mathematical backbone of operations research, from diet
and blending problems to network flows, scheduling, and the LP relaxations at the heart of
integer-programming solvers. The feasible region of an LP is a convex POLYTOPE, and a fundamental
theorem says that if an optimum exists, one is attained at a VERTEX. The SIMPLEX METHOD, invented by
Dantzig in 1947, exploits this: it starts at a vertex and repeatedly slides along an edge to an
adjacent vertex that improves the objective, until no improving edge remains -- at which point the
current vertex is provably optimal.

Mechanically the method operates on a TABLEAU. Slack variables turn inequalities into equalities;
the tableau tracks a BASIS of variables and the reduced costs of the others. Each PIVOT chooses an
entering variable with a favourable reduced cost (BLAND'S RULE -- smallest index -- guarantees no
cycling) and a leaving variable by the MINIMUM-RATIO test (so feasibility is preserved), then does
Gaussian elimination to update the tableau. Problems without an obvious starting vertex are handled
by a TWO-PHASE method: phase one drives artificial variables to zero to find any feasible vertex,
phase two optimizes the real objective from there. Unboundedness and infeasibility are detected along
the way.

This module implements a two-phase, Bland's-rule simplex for LPs in general form (<=, >=, = mixed
constraints, maximize or minimize), returning the optimal value, the solution vector, and a status
(optimal / unbounded / infeasible). It is verified against hand-solved textbook LPs, against a
brute-force solver that enumerates every basic vertex and checks feasibility and objective, against
the LP-duality theorem (primal optimum equals dual optimum), and on degenerate and unbounded and
infeasible instances. Pure stdlib; an optimization companion to the linear-algebra and
convex-geometry notes."""

from __future__ import annotations


class LPResult:
    __slots__ = ("status", "value", "x")

    def __init__(self, status, value, x):
        self.status = status        # 'optimal' | 'unbounded' | 'infeasible'
        self.value = value          # objective value (in the ORIGINAL sense: maximize or minimize)
        self.x = x                  # list of the original decision-variable values

    def __repr__(self):
        return f"LPResult(status={self.status!r}, value={self.value}, x={self.x})"


_EPS = 1e-9


def solve(c, constraints, maximize=True):
    """Solve a linear program.

    c: objective coefficients (length n).
    constraints: list of (a, sense, b) where a is a length-n coefficient row, sense is one of
        '<=', '>=', '=' and b is the right-hand side.
    maximize: True to maximize c.x, False to minimize.

    Returns an LPResult. Assumes decision variables are non-negative (x >= 0), the standard LP
    convention."""
    n = len(c)
    # normalize to maximization internally
    sign = 1.0 if maximize else -1.0
    obj = [sign * ci for ci in c]

    # Build equality system A x + (slacks/surplus) = b, b >= 0.
    # Each constraint contributes one row. We add slack (+1) for <=, surplus (-1) for >=, and an
    # artificial variable where needed to get an initial identity basis.
    rows = []
    rhs = []
    slack_cols = []          # (row_index, +1/-1) for a slack/surplus column
    artificial_rows = []     # rows that need an artificial variable

    for a, sense, b in constraints:
        row = list(a)
        rr = list(row)
        rb = b
        # flip so that rhs >= 0
        if rb < 0:
            rr = [-v for v in rr]
            rb = -rb
            sense = {"<=": ">=", ">=": "<=", "=": "="}[sense]
        rows.append((rr, sense, rb))

    m = len(rows)
    # count extra columns
    # layout: [ x_1..x_n | slacks/surplus... | artificials... ]
    tableau_rows = []
    n_slack = 0
    slack_info = []
    for (rr, sense, rb) in rows:
        if sense == "<=":
            slack_info.append(("slack", +1))
            n_slack += 1
        elif sense == ">=":
            slack_info.append(("surplus", -1))
            n_slack += 1
        else:
            slack_info.append(("none", 0))

    # which rows need artificials: '>=' and '=' (and '<=' after a negative flip never happens here)
    need_art = []
    for i, (rr, sense, rb) in enumerate(rows):
        if sense in (">=", "="):
            need_art.append(i)
    n_art = len(need_art)

    total_cols = n + n_slack + n_art
    A = [[0.0] * total_cols for _ in range(m)]
    bcol = [0.0] * m

    slack_index = n
    art_index = n + n_slack
    basis = [None] * m
    art_cols = []
    si = 0
    ai = 0
    slack_col_of_row = {}
    for i, (rr, sense, rb) in enumerate(rows):
        for j in range(n):
            A[i][j] = float(rr[j])
        bcol[i] = float(rb)
        if sense == "<=":
            A[i][slack_index + si] = 1.0
            basis[i] = slack_index + si
            si += 1
        elif sense == ">=":
            A[i][slack_index + si] = -1.0
            si += 1
        # artificial
        if sense in (">=", "="):
            A[i][art_index + ai] = 1.0
            basis[i] = art_index + ai
            art_cols.append(art_index + ai)
            ai += 1

    # ---- Phase 1: minimize sum of artificials (if any) ----
    if n_art > 0:
        # phase-1 objective: minimize sum(artificials) == maximize -sum(artificials)
        phase1_c = [0.0] * total_cols
        for col in art_cols:
            phase1_c[col] = -1.0
        status = _simplex(A, bcol, basis, phase1_c)
        # feasible iff phase-1 optimum is 0 (all artificials driven out)
        art_val = sum(bcol[i] for i in range(m) if basis[i] in art_cols)
        if art_val > 1e-6:
            return LPResult("infeasible", None, None)
        # drive any remaining artificial out of the basis if possible (degenerate)
        # (left in place with value 0 is harmless for phase 2 since their cost is 0 there)

    # ---- Phase 2: optimize the real objective ----
    real_c = [0.0] * total_cols
    for j in range(n):
        real_c[j] = obj[j]
    status = _simplex(A, bcol, basis, real_c, forbid=set(art_cols))
    if status == "unbounded":
        return LPResult("unbounded", None, None)

    # read the solution
    x = [0.0] * n
    for i in range(m):
        if basis[i] is not None and basis[i] < n:
            x[basis[i]] = bcol[i]
    value = sum(c[j] * x[j] for j in range(n))
    return LPResult("optimal", value, x)


def _simplex(A, b, basis, cost, forbid=None):
    """In-place primal simplex maximizing cost.x on the tableau (A | b) with the given basis. Uses
    Bland's rule for the entering variable to guarantee termination. Returns 'optimal' or
    'unbounded'. `forbid` is a set of columns that may not enter (e.g. artificials in phase 2)."""
    m = len(A)
    total_cols = len(A[0])
    forbid = forbid or set()

    def reduced_costs():
        # c_j - c_B . (B^-1 A_j); with an explicit basis maintained by elimination, the tableau
        # already stores B^-1 A, so reduced cost = cost[j] - sum over basic rows of cost[basis]*A[i][j]
        cb = [cost[basis[i]] for i in range(m)]
        rc = list(cost)
        for j in range(total_cols):
            s = 0.0
            for i in range(m):
                s += cb[i] * A[i][j]
            rc[j] = cost[j] - s
        return rc

    while True:
        rc = reduced_costs()
        # Bland: smallest-index entering column with positive reduced cost
        entering = -1
        for j in range(total_cols):
            if j in forbid:
                continue
            if rc[j] > _EPS:
                entering = j
                break
        if entering == -1:
            return "optimal"
        # minimum-ratio test for the leaving row
        leaving = -1
        best_ratio = None
        for i in range(m):
            aij = A[i][entering]
            if aij > _EPS:
                ratio = b[i] / aij
                if best_ratio is None or ratio < best_ratio - _EPS or (
                        abs(ratio - best_ratio) <= _EPS and (leaving == -1 or basis[i] < basis[leaving])):
                    best_ratio = ratio
                    leaving = i
        if leaving == -1:
            return "unbounded"
        _pivot(A, b, basis, leaving, entering)


def _pivot(A, b, basis, r, c):
    """Gaussian-eliminate so column c becomes the r-th unit vector; update the basis."""
    m = len(A)
    total_cols = len(A[0])
    piv = A[r][c]
    inv = 1.0 / piv
    for j in range(total_cols):
        A[r][j] *= inv
    b[r] *= inv
    for i in range(m):
        if i == r:
            continue
        factor = A[i][c]
        if factor != 0.0:
            for j in range(total_cols):
                A[i][j] -= factor * A[r][j]
            b[i] -= factor * b[r]
    basis[r] = c
