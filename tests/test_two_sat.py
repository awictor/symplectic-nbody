"""Tests for two_sat: verdict + assignment vs brute force over all 2^n assignments."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from two_sat import TwoSAT

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 909
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_satisfiable(n, clauses):
    for bits in itertools.product([False, True], repeat=n):
        ok = True
        for a, b in clauses:
            va = bits[abs(a) - 1] if a > 0 else not bits[abs(a) - 1]
            vb = bits[abs(b) - 1] if b > 0 else not bits[abs(b) - 1]
            if not (va or vb):
                ok = False
                break
        if ok:
            return True
    return False


# --- a known satisfiable formula -------------------------------------------
s = TwoSAT(3)
s.add_clause(1, 2)
s.add_clause(-1, 2)
s.add_clause(-2, 3)
a = s.solve()
check("known satisfiable formula returns an assignment", a is not None)
check("returned assignment satisfies the formula", a is not None and s.check(a))

# --- canonical contradiction (x) AND (not x) -------------------------------
u = TwoSAT(1)
u.force_true(1)
u.force_true(-1)
check("(x) AND (not x) is unsatisfiable", u.solve() is None)
check("is_satisfiable agrees", not u.is_satisfiable())

# --- the classic 4-clause unsatisfiable formula ----------------------------
u2 = TwoSAT(2)
for c in [(1, 2), (1, -2), (-1, 2), (-1, -2)]:
    u2.add_clause(*c)
check("(a|b)(a|~b)(~a|b)(~a|~b) is unsatisfiable", u2.solve() is None)

# --- verdict matches brute force over many random formulas -----------------
verdict_ok = True
assign_ok = True
for _ in range(300):
    n = 2 + int(rng() * 5)            # 2..6 variables
    m = 1 + int(rng() * n * 3)
    clauses = []
    for _ in range(m):
        a = (int(rng() * n) + 1) * (1 if rng() < 0.5 else -1)
        b = (int(rng() * n) + 1) * (1 if rng() < 0.5 else -1)
        clauses.append((a, b))
    solver = TwoSAT(n)
    for c in clauses:
        solver.add_clause(*c)
    result = solver.solve()
    brute = brute_satisfiable(n, clauses)
    if (result is not None) != brute:
        verdict_ok = False
        break
    if result is not None and not solver.check(result):
        assign_ok = False
        break
check("satisfiability verdict matches brute force over 300 random formulas", verdict_ok)
check("every returned assignment satisfies all clauses", assign_ok)

# --- implications ----------------------------------------------------------
# x1 -> x2, x2 -> x3, and x1 forced true => all true
s = TwoSAT(3)
s.add_implication(1, 2)
s.add_implication(2, 3)
s.force_true(1)
a = s.solve()
check("implication chain forces all true", a == [True, True, True])

# --- a chain that forces a contradiction -----------------------------------
# x1 true, x1 -> x2, x2 -> not x1  => contradiction
s = TwoSAT(2)
s.force_true(1)
s.add_implication(1, 2)
s.add_implication(2, -1)
check("contradictory implication chain is unsatisfiable", s.solve() is None)

# --- a single free variable ------------------------------------------------
s = TwoSAT(1)
check("no clauses -> trivially satisfiable", s.solve() is not None)

# --- larger satisfiable instance -------------------------------------------
big = TwoSAT(50)
# a satisfiable random-ish instance: build from a known assignment
truth = [rng() < 0.5 for _ in range(50)]
for _ in range(120):
    # each clause includes at least one literal true under `truth` -> guaranteed satisfiable
    i = int(rng() * 50)
    lit_true = (i + 1) if truth[i] else -(i + 1)
    j = int(rng() * 50)
    lit_other = (j + 1) * (1 if rng() < 0.5 else -1)
    big.add_clause(lit_true, lit_other)
a = big.solve()
check("larger satisfiable instance is solved", a is not None and big.check(a))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all two_sat tests passed")
