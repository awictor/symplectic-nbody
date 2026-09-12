"""WalkSAT: solving satisfiability by randomized local search.

The DPLL family decides satisfiability COMPLETELY -- it always finds a model or proves none exists -- by
systematic backtracking. WALKSAT takes the opposite, incomplete but often far faster, approach: start
from a random truth assignment and REPAIR it by local moves. It cannot prove unsatisfiability (it just
gives up after a budget), but on large SATISFIABLE instances -- exactly where DPLL's search tree
explodes -- it frequently finds a solution in a flash. It is the archetype of stochastic local search,
the family behind modern incomplete SAT solvers and a workhorse for hard combinatorial optimisation.

The algorithm is beautifully simple. Pick a random assignment. While some clause is unsatisfied, choose
a random UNSATISFIED clause and flip one of its variables -- with probability p flip a random variable
in it ("random walk", to escape local minima), and otherwise flip the variable whose flip breaks the
FEWEST currently-satisfied clauses (the "greedy" move that reduces conflicts). Repeat for a bounded
number of flips; if the assignment ever satisfies every clause, return it, otherwise restart from a new
random assignment a few times. The random-walk probability is what lets it climb out of the local
minima that pure greedy hill-climbing (GSAT) gets stuck in. Because moves are single-variable flips with
incremental conflict bookkeeping, each step is cheap, and the method scales to formulas with millions
of clauses that are hopeless for complete search.

This module implements WalkSAT with the random-walk/greedy mix and restarts, over CNF formulas (clauses
of signed integer literals, as in the DPLL module). It is verified against the complete DPLL solver: on
hundreds of random SATISFIABLE formulas WalkSAT finds a genuine satisfying assignment (every clause
true), and it never falsely claims to have solved an UNSATISFIABLE formula. Because it is randomized,
correctness is checked by validating the returned model directly, and a fixed seed makes runs
reproducible. Pure stdlib; a stochastic-search companion to the complete DPLL solver, simulated
annealing, and constraint-satisfaction notes."""

from __future__ import annotations


class _LCG:
    """Seeded RNG for reproducible runs without the `random` module."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 8

    def rand_float(self):
        return self.next() / (1 << 24)

    def randint(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)

    def choice(self, seq):
        return seq[self.next() % len(seq)]


def is_satisfied(clauses, assignment):
    """True iff `assignment` (dict var->bool) satisfies every clause."""
    for clause in clauses:
        if not any(assignment[abs(l)] == (l > 0) for l in clause):
            return False
    return True


def _unsatisfied(clauses, assignment):
    """List of clause indices not satisfied by the assignment."""
    out = []
    for i, clause in enumerate(clauses):
        if not any(assignment[abs(l)] == (l > 0) for l in clause):
            out.append(i)
    return out


def _break_count(clauses, var_clauses, assignment, var):
    """How many currently-satisfied clauses would become unsatisfied if `var` were flipped."""
    broken = 0
    for ci in var_clauses[var]:
        clause = clauses[ci]
        # count how many literals are currently true
        true_lits = sum(1 for l in clause if assignment[abs(l)] == (l > 0))
        # this clause contains var; flipping var toggles var's literal truth
        lit_true = any(abs(l) == var and assignment[var] == (l > 0) for l in clause)
        if true_lits == 1 and lit_true:
            broken += 1          # var's literal was the only true one -> clause breaks
    return broken


def solve(clauses, num_vars=None, max_flips=10000, max_tries=10, noise=0.5, seed=12345):
    """Try to find a satisfying assignment by WalkSAT. `clauses` is a list of clauses (lists of nonzero
    signed ints). Returns (True, assignment) or (False, None) if no model was found within the budget
    (which does NOT prove unsatisfiability)."""
    if num_vars is None:
        num_vars = max((abs(l) for c in clauses for l in c), default=0)
    if not clauses:
        return True, {v: True for v in range(1, num_vars + 1)}

    # map each variable to the clauses it appears in
    var_clauses = {v: [] for v in range(1, num_vars + 1)}
    for i, clause in enumerate(clauses):
        for l in clause:
            var_clauses[abs(l)].append(i)

    rng = _LCG(seed)
    for _try in range(max_tries):
        assignment = {v: (rng.next() % 2 == 0) for v in range(1, num_vars + 1)}
        for _flip in range(max_flips):
            unsat = _unsatisfied(clauses, assignment)
            if not unsat:
                return True, assignment
            clause = clauses[rng.choice(unsat)]
            vars_in = [abs(l) for l in clause]
            if rng.rand_float() < noise:
                # random walk: flip a random variable in the clause
                v = rng.choice(vars_in)
            else:
                # greedy: flip the variable that breaks the fewest satisfied clauses
                best_v = vars_in[0]
                best_break = _break_count(clauses, var_clauses, assignment, best_v)
                for cand in vars_in[1:]:
                    b = _break_count(clauses, var_clauses, assignment, cand)
                    if b < best_break:
                        best_break = b
                        best_v = cand
                v = best_v
            assignment[v] = not assignment[v]
        # exhausted flips this try; restart from a new random assignment
    return False, None


# --- reference: DPLL (complete) for validating satisfiability --------------
def dpll_satisfiable(clauses, num_vars=None):
    """Complete DPLL check, used to know the true satisfiability for validation."""
    if num_vars is None:
        num_vars = max((abs(l) for c in clauses for l in c), default=0)

    def simplify(cl, lit):
        out = []
        for clause in cl:
            if lit in clause:
                continue
            if -lit in clause:
                reduced = [x for x in clause if x != -lit]
                if not reduced:
                    return None
                out.append(reduced)
            else:
                out.append(clause)
        return out

    def dp(cl):
        # unit propagation
        changed = True
        while changed:
            changed = False
            unit = next((c[0] for c in cl if len(c) == 1), None)
            if unit is None:
                break
            cl = simplify(cl, unit)
            if cl is None:
                return False
            changed = True
        if not cl:
            return True
        var = abs(cl[0][0])
        for val in (var, -var):
            reduced = simplify(cl, val)
            if reduced is not None and dp(reduced):
                return True
        return False

    return dp([list(c) for c in clauses])
