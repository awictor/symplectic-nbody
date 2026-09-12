"""Tests for walksat: randomized local-search SAT validated against complete DPLL."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from walksat import solve, is_satisfied, dpll_satisfiable

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


def random_cnf(rng, n, m, k=3):
    clauses = []
    for _ in range(m):
        clause = []
        for _ in range(k):
            v = rng.randint(1, n)
            clause.append(v if rng.rand() % 2 else -v)
        clauses.append(clause)
    return clauses


# --- known cases ------------------------------------------------------------
c = [[1, 2, -3], [-1, 3], [-2, 3], [1, -2], [2, 3]]
sat, a = solve(c)
check("satisfiable formula is solved", sat)
check("returned model actually satisfies it", is_satisfied(c, a))

check("empty formula is trivially satisfiable", solve([])[0])
check("a unit clause is satisfiable with the forced value",
      is_satisfied([[5]], solve([[5]])[1]))

# --- WalkSAT solves satisfiable instances (checked by DPLL) ----------------
rng = LCG(2026)
solved = tested = 0
model_valid = True
for _ in range(200):
    n = rng.randint(3, 20)
    m = rng.randint(1, int(n * 3.5))       # ratio below the phase transition -> mostly SAT
    clauses = random_cnf(rng, n, m)
    if not dpll_satisfiable(clauses, n):
        continue
    tested += 1
    sat, model = solve(clauses, n, max_flips=3000, max_tries=30, seed=rng.randint(1, 10 ** 9))
    if sat:
        solved += 1
        if not is_satisfied(clauses, model):
            model_valid = False
            break
check("every model WalkSAT returns genuinely satisfies the formula", model_valid)
# WalkSAT is incomplete but should solve the vast majority of satisfiable instances at low ratio
check(f"WalkSAT solves the overwhelming majority of satisfiable instances ({solved}/{tested})",
      tested > 0 and solved >= 0.95 * tested)

# --- WalkSAT never falsely claims to solve an UNSAT formula ----------------
rng = LCG(4242)
no_false_ok = True
unsat_seen = 0
for _ in range(200):
    n = rng.randint(3, 8)
    m = rng.randint(n * 4, n * 7)          # high ratio -> often UNSAT
    clauses = random_cnf(rng, n, m)
    if dpll_satisfiable(clauses, n):
        continue
    unsat_seen += 1
    sat, model = solve(clauses, n, max_flips=500, max_tries=5, seed=rng.randint(1, 10 ** 9))
    # WalkSAT must NOT claim success on an unsatisfiable formula
    if sat:
        no_false_ok = False
        break
check("WalkSAT never claims to solve an unsatisfiable formula", no_false_ok)
check("random suite produced unsatisfiable formulas to test against", unsat_seen > 0)

# --- reproducibility --------------------------------------------------------
clauses = random_cnf(LCG(1), 15, 45)
r1 = solve(clauses, 15, seed=999)
r2 = solve(clauses, 15, seed=999)
check("same seed gives the same result", r1[0] == r2[0] and (not r1[0] or r1[1] == r2[1]))

# --- a forced chain (unit propagation would nail it; WalkSAT must too) -----
chain = [[1], [-1, 2], [-2, 3], [-3, 4], [-4, 5]]
sat, model = solve(chain, 5, max_flips=2000, max_tries=20)
check("forced chain x1->..->x5 is solved with all true",
      sat and all(model[v] for v in range(1, 6)))

# --- larger satisfiable instance (planted solution) ------------------------
# build clauses that are all satisfied by a known random assignment
rng = LCG(31337)
n = 50
planted = {v: (rng.rand() % 2 == 0) for v in range(1, n + 1)}
clauses = []
for _ in range(200):
    clause = []
    for _ in range(3):
        v = rng.randint(1, n)
        # bias so the clause is satisfied by `planted` (at least one literal true)
        clause.append(v if planted[v] else -v)
    clauses.append(clause)
sat, model = solve(clauses, n, max_flips=5000, max_tries=30)
check("50-var planted-satisfiable instance is solved with a valid model",
      sat and is_satisfied(clauses, model))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all walksat tests passed")
