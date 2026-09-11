"""Tests for simplex: textbook LPs, brute-force vertex enumeration, LP duality, edge cases."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simplex import solve

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 4242
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- Gaussian solve for the brute-force vertex enumerator ------------------
def gauss_solve(M, y):
    n = len(M)
    A = [row[:] + [y[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            return None
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [v / pv for v in A[col]]
        for r in range(n):
            if r != col and abs(A[r][col]) > 1e-15:
                f = A[r][col]
                A[r] = [A[r][k] - f * A[col][k] for k in range(n + 1)]
    return [A[i][n] for i in range(n)]


def brute_lp(c, constraints, maximize=True):
    """Enumerate every vertex (intersection of n active constraints, including x_i>=0) and take the
    best feasible one. Returns (status, value)."""
    n = len(c)
    # all constraint rows as (a, sense, b), plus the non-negativity constraints x_i >= 0
    rows = [(list(a), sense, b) for a, sense, b in constraints]
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        rows.append((e, ">=", 0.0))

    def feasible(x):
        for a, sense, b in rows:
            lhs = sum(a[j] * x[j] for j in range(n))
            if sense == "<=" and lhs > b + 1e-6:
                return False
            if sense == ">=" and lhs < b - 1e-6:
                return False
            if sense == "=" and abs(lhs - b) > 1e-6:
                return False
        return True

    best = None
    for combo in itertools.combinations(range(len(rows)), n):
        M = [rows[i][0] for i in combo]
        y = [rows[i][2] for i in combo]
        x = gauss_solve(M, y)
        if x is None:
            continue
        if feasible(x):
            val = sum(c[j] * x[j] for j in range(n))
            if best is None or (maximize and val > best) or (not maximize and val < best):
                best = val
    if best is None:
        return ("infeasible", None)
    return ("optimal", best)


# --- textbook LP -----------------------------------------------------------
# maximize 3x + 5y  s.t.  x <= 4, 2y <= 12, 3x + 2y <= 18, x,y>=0.  Optimum 36 at (2, 6).
r = solve([3, 5], [([1, 0], "<=", 4), ([0, 2], "<=", 12), ([3, 2], "<=", 18)], maximize=True)
check("textbook LP status optimal", r.status == "optimal")
check("textbook LP optimum is 36", abs(r.value - 36) < 1e-6)
check("textbook LP solution is (2,6)", abs(r.x[0] - 2) < 1e-6 and abs(r.x[1] - 6) < 1e-6)

# --- a minimization --------------------------------------------------------
# minimize 2x + 3y  s.t.  x + y >= 10, x >= 0, y >= 0.  Optimum 20 at (10, 0).
r = solve([2, 3], [([1, 1], ">=", 10)], maximize=False)
check("min LP optimum is 20", abs(r.value - 20) < 1e-6)

# --- equality constraint ---------------------------------------------------
# maximize x + y s.t. x + y = 5, x <= 3.  Optimum 5 (any split); value must be 5.
r = solve([1, 1], [([1, 1], "=", 5), ([1, 0], "<=", 3)], maximize=True)
check("equality-constrained LP optimum is 5", r.status == "optimal" and abs(r.value - 5) < 1e-6)

# --- unbounded -------------------------------------------------------------
# maximize x + y s.t. x - y <= 1  (unbounded above)
r = solve([1, 1], [([1, -1], "<=", 1)], maximize=True)
check("unbounded LP detected", r.status == "unbounded")

# --- infeasible ------------------------------------------------------------
# x >= 5 and x <= 2 : infeasible
r = solve([1], [([1], ">=", 5), ([1], "<=", 2)], maximize=True)
check("infeasible LP detected", r.status == "infeasible")

# --- brute-force cross-check over random feasible LPs ----------------------
matches = 0
trials = 0
for _ in range(60):
    n = 2 + int(rng() * 2)          # 2 or 3 variables
    mrows = 2 + int(rng() * 3)
    c = [round(rng() * 6 - 1, 1) for _ in range(n)]
    cons = []
    for _ in range(mrows):
        a = [round(rng() * 4, 1) for _ in range(n)]
        if all(v == 0 for v in a):
            a[0] = 1.0
        b = round(2 + rng() * 12, 1)
        cons.append((a, "<=", b))     # all <= with positive b -> always feasible (origin works),
        # and bounded since coefficients/rhs are positive and we maximize non-negative-ish c
    maximize = True
    r = solve(c, cons, maximize=maximize)
    bstatus, bval = brute_lp(c, cons, maximize=maximize)
    trials += 1
    if r.status == "optimal" and bstatus == "optimal":
        if abs(r.value - bval) < 1e-4:
            matches += 1
    elif r.status == bstatus:
        matches += 1
check(f"simplex matches brute-force vertex enumeration ({matches}/{trials})", matches == trials)

# --- LP duality: primal max == dual min ------------------------------------
# primal: max c.x s.t. Ax <= b, x>=0.  dual: min b.y s.t. A^T y >= c, y>=0.
def dual_check(c, A, b):
    primal = solve(c, [(A[i], "<=", b[i]) for i in range(len(A))], maximize=True)
    n = len(c)
    m = len(A)
    AT = [[A[i][j] for i in range(m)] for j in range(n)]
    dual = solve(b, [(AT[j], ">=", c[j]) for j in range(n)], maximize=False)
    return primal, dual

p, d = dual_check([3, 5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18])
check("LP duality: primal optimum equals dual optimum",
      p.status == "optimal" and d.status == "optimal" and abs(p.value - d.value) < 1e-5)

p, d = dual_check([2, 3, 1], [[1, 1, 1], [2, 1, 0], [0, 1, 3]], [10, 8, 15])
check("LP duality holds on a 3-variable LP",
      p.status == "optimal" and d.status == "optimal" and abs(p.value - d.value) < 1e-5)

# --- degeneracy: Bland's rule must not cycle -------------------------------
# a classic degenerate LP
r = solve([10, -57, -9, -24],
          [([0.5, -5.5, -2.5, 9], "<=", 0),
           ([0.5, -1.5, -0.5, 1], "<=", 0),
           ([1, 0, 0, 0], "<=", 1)], maximize=True)
check("degenerate LP terminates (no cycling) and is optimal", r.status == "optimal")
check("degenerate LP optimum is 1 (known)", abs(r.value - 1.0) < 1e-6)

# --- solution actually satisfies the constraints ---------------------------
r = solve([4, 3], [([2, 3], "<=", 6), ([-3, 2], "<=", 3), ([0, 2], "<=", 5), ([2, 1], "<=", 4)],
          maximize=True)
ok = True
for a, sense, b in [([2, 3], "<=", 6), ([-3, 2], "<=", 3), ([0, 2], "<=", 5), ([2, 1], "<=", 4)]:
    lhs = sum(a[j] * r.x[j] for j in range(2))
    if lhs > b + 1e-6:
        ok = False
check("simplex solution satisfies all constraints", r.status == "optimal" and ok)
check("simplex solution is non-negative", all(v >= -1e-9 for v in r.x))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all simplex tests passed")
