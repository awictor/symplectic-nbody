"""DPLL: deciding Boolean satisfiability by backtracking search with propagation.

The BOOLEAN SATISFIABILITY problem (SAT) asks whether a formula in conjunctive normal form -- an AND of
clauses, each clause an OR of literals (a variable or its negation) -- can be made true by some
assignment of true/false to its variables. It is the archetypal NP-complete problem: circuit
verification, planning, dependency resolution, and countless combinatorial puzzles all compile down to
SAT, and a fast SAT decision answers them all. Brute force tries all 2^n assignments; the
DAVIS-PUTNAM-LOGEMANN-LOVELAND (DPLL) procedure, from 1962, is the backtracking search at the heart of
every modern SAT solver, and in practice explores a tiny fraction of that space.

DPLL is depth-first assignment with two propagation rules that prune enormously. UNIT PROPAGATION: if a
clause has all but one literal already false, that last literal MUST be true, so assign it and cascade
-- one forced assignment often triggers a chain of others, and a clause whose every literal is false
means the current branch is dead (a conflict). PURE LITERAL elimination: if a variable appears with
only one polarity across all remaining clauses, assigning it that way can never hurt, so fix it and
drop every clause it satisfies. After propagation, DPLL picks an unassigned variable, tries it true,
and recurses; if that fails it tries false; if both fail the formula is unsatisfiable under the current
partial assignment. The empty clause set means SAT (everything satisfied); a formula containing an
empty clause means a conflict.

This module parses a CNF as a list of clauses (each a list of nonzero signed integers, +v for variable
v true, -v for false), decides satisfiability with DPLL plus unit propagation and pure-literal
elimination, and returns a satisfying assignment when one exists. It is verified against a brute-force
truth-table oracle -- for every random formula, DPLL reports SAT exactly when some assignment satisfies
it, and every model DPLL returns genuinely satisfies all clauses -- and on hand-built cases including
the pigeonhole principle (n+1 pigeons in n holes is unsatisfiable). Pure stdlib; a logic-and-search
companion to the 2-SAT and combinatorial-search notes."""

from __future__ import annotations


def _simplify(clauses, lit):
    """Assign `lit` true: drop every clause it satisfies, and remove -lit from the rest. Returns the
    new clause list, or None if this produces an empty clause (a conflict)."""
    out = []
    for clause in clauses:
        if lit in clause:
            continue                     # clause satisfied, drop it
        if -lit in clause:
            reduced = [x for x in clause if x != -lit]
            if not reduced:
                return None              # emptied a clause -> conflict
            out.append(reduced)
        else:
            out.append(clause)
    return out


def _unit_propagate(clauses, assignment):
    """Repeatedly assign forced unit-clause literals. Returns (clauses, ok): ok is False on
    conflict. Mutates `assignment` in place with the forced values."""
    changed = True
    while changed:
        changed = False
        unit = None
        for clause in clauses:
            if len(clause) == 1:
                unit = clause[0]
                break
        if unit is None:
            break
        assignment[abs(unit)] = (unit > 0)
        clauses = _simplify(clauses, unit)
        if clauses is None:
            return None, False
        changed = True
    return clauses, True


def _pure_literal_assign(clauses, assignment):
    """Fix every pure literal (a variable appearing with only one polarity). Returns the reduced
    clause list; records the fixed values in `assignment`."""
    counts = {}
    for clause in clauses:
        for lit in clause:
            counts[lit] = counts.get(lit, 0) + 1
    pures = [lit for lit in counts if -lit not in counts]
    for lit in pures:
        assignment[abs(lit)] = (lit > 0)
        clauses = _simplify(clauses, lit)
        if clauses is None:              # cannot happen for a pure literal, but stay safe
            return None
    return clauses


def solve(clauses, num_vars=None):
    """Decide satisfiability of a CNF formula by DPLL.

    `clauses` is a list of clauses; each clause is a list of nonzero ints (+v means variable v is
    true, -v means false). Returns (True, assignment) with assignment a dict {var: bool} covering
    every variable if satisfiable, or (False, None) if unsatisfiable. Free variables (unconstrained)
    are reported as True."""
    if num_vars is None:
        num_vars = max((abs(l) for c in clauses for l in c), default=0)
    # normalise clauses to lists (and copy so we never mutate the caller's data)
    clauses = [list(c) for c in clauses]

    def dpll(clauses, assignment):
        clauses, ok = _unit_propagate(clauses, assignment)
        if not ok:
            return None
        clauses = _pure_literal_assign(clauses, assignment)
        if clauses is None:
            return None
        if not clauses:
            return assignment            # all clauses satisfied
        # choose a branching variable from the first remaining clause
        var = abs(clauses[0][0])
        for value in (True, False):
            lit = var if value else -var
            branch = dict(assignment)
            branch[var] = value
            reduced = _simplify(clauses, lit)
            if reduced is None:
                continue
            result = dpll(reduced, branch)
            if result is not None:
                return result
        return None

    result = dpll(clauses, {})
    if result is None:
        return False, None
    # fill in any variables never forced (unconstrained) as True
    full = {v: result.get(v, True) for v in range(1, num_vars + 1)}
    return True, full


def is_satisfied(clauses, assignment):
    """True iff `assignment` (a dict {var: bool}) satisfies every clause."""
    for clause in clauses:
        if not any((assignment.get(abs(lit), lit > 0)) == (lit > 0) for lit in clause):
            return False
    return True


# --- brute-force reference --------------------------------------------------
def brute_satisfiable(clauses, num_vars=None):
    """Decide SAT by enumerating all 2^n assignments. Returns (sat, assignment_or_None). Exponential;
    for tests on small formulas only."""
    if num_vars is None:
        num_vars = max((abs(l) for c in clauses for l in c), default=0)
    for mask in range(1 << num_vars):
        assignment = {v: bool((mask >> (v - 1)) & 1) for v in range(1, num_vars + 1)}
        if is_satisfied(clauses, assignment):
            return True, assignment
    return False, None


def pigeonhole(n):
    """The pigeonhole CNF: place n+1 pigeons into n holes, no two pigeons sharing a hole.
    Unsatisfiable for all n >= 1. Variable x(p,h) = pigeon p is in hole h, encoded as
    p*n + h + 1 for p in 0..n, h in 0..n-1."""
    def var(p, h):
        return p * n + h + 1

    clauses = []
    # each pigeon is in at least one hole
    for p in range(n + 1):
        clauses.append([var(p, h) for h in range(n)])
    # no two pigeons share a hole
    for h in range(n):
        for p1 in range(n + 1):
            for p2 in range(p1 + 1, n + 1):
                clauses.append([-var(p1, h), -var(p2, h)])
    return clauses, (n + 1) * n
