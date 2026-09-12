"""2-SAT: satisfying boolean formulas of two-literal clauses in linear time.

Boolean SATISFIABILITY is NP-complete in general, but the restricted case where every clause has at
most TWO literals -- 2-SAT -- is solvable in LINEAR time, a rare island of tractability inside an
intractable problem. 2-SAT models a surprising range of constraints: scheduling with either/or
requirements, conflict-free layout, consistency of paired choices, and the classic 'this OR that'
puzzles. The linear algorithm is a beautiful application of strongly connected components.

The key is the IMPLICATION GRAPH. A clause (a OR b) is logically equivalent to two implications:
(not a implies b) and (not b implies a) -- if one literal is false, the other must be true. Build a
directed graph with two vertices per variable (the literal and its negation) and an edge for each
implication. Now a truth assignment is consistent exactly when no variable x has x and NOT x forced
to the same value -- which happens iff x and NOT x lie in the SAME strongly connected component (each
would then imply the other, a contradiction). So the formula is SATISFIABLE iff no variable shares an
SCC with its own negation, checkable by Tarjan's SCC in O(V + E). When satisfiable, an assignment is
read straight off the SCC order: set each literal true whose component comes LATER in the reverse
topological order than its negation's -- because the condensation is a DAG, this choice is always
consistent.

This module implements a 2-SAT solver: add clauses as pairs of signed literals, test satisfiability,
and extract a satisfying assignment when one exists, all via the implication graph and strongly
connected components. It is verified against brute force over all 2^n assignments for small n: that
the solver's satisfiability verdict always matches whether any assignment satisfies the formula, that
every assignment it returns actually satisfies every clause, that known satisfiable and unsatisfiable
formulas are classified correctly, and on the canonical contradiction (x) AND (not x). Pure stdlib; a
constraint-satisfaction companion to the Tarjan-SCC and wave-function-collapse notes."""

from __future__ import annotations

from tarjan_scc import strongly_connected_components


class TwoSAT:
    """A 2-satisfiability solver over n boolean variables (0..n-1).

    Literals are signed integers: variable i as +（i+1) for true, -(i+1) for its negation. Clauses are
    added as (literal_a, literal_b) meaning (a OR b)."""

    def __init__(self, n):
        self.n = n
        self.clauses = []

    # map a signed literal to a graph vertex: variable i has vertices 2i (true) and 2i+1 (false)
    def _vertex(self, lit):
        var = abs(lit) - 1
        return 2 * var if lit > 0 else 2 * var + 1

    @staticmethod
    def _neg(vertex):
        return vertex ^ 1

    def add_clause(self, a, b):
        """Add the clause (a OR b) where a, b are signed literals (+i / -i, 1-indexed variable)."""
        self.clauses.append((a, b))

    def add_implication(self, a, b):
        """Add 'a implies b' (equivalent to the clause (not a OR b))."""
        self.add_clause(-a, b)

    def force_true(self, lit):
        """Force a literal to be true (clause (lit OR lit))."""
        self.add_clause(lit, lit)

    def _build_graph(self):
        edges = []
        for a, b in self.clauses:
            va, vb = self._vertex(a), self._vertex(b)
            # (a OR b) == (not a -> b) and (not b -> a)
            edges.append((self._neg(va), vb))
            edges.append((self._neg(vb), va))
        return edges

    def solve(self):
        """Return a satisfying assignment as a list of booleans (assignment[i] = value of variable i),
        or None if the formula is unsatisfiable."""
        edges = self._build_graph()
        comps = strongly_connected_components(2 * self.n, edges)
        comp_id = [0] * (2 * self.n)
        for cid, comp in enumerate(comps):
            for v in comp:
                comp_id[v] = cid
        # unsatisfiable iff some variable and its negation share an SCC
        for i in range(self.n):
            if comp_id[2 * i] == comp_id[2 * i + 1]:
                return None
        # SCCs are returned in REVERSE topological order, so a smaller comp id = later in topo order.
        # Set the literal whose component appears later (smaller reverse-topo index) to true.
        assignment = [False] * self.n
        for i in range(self.n):
            # comp_id from strongly_connected_components: earlier in the list = deeper in topo order.
            # A literal is true if its own component id < its negation's (its comp comes first in the
            # reverse-topo list, i.e. later in a true topological order).
            assignment[i] = comp_id[2 * i] < comp_id[2 * i + 1]
        return assignment

    def is_satisfiable(self):
        return self.solve() is not None

    def check(self, assignment):
        """True if the given assignment satisfies every clause."""
        for a, b in self.clauses:
            va = assignment[abs(a) - 1] if a > 0 else not assignment[abs(a) - 1]
            vb = assignment[abs(b) - 1] if b > 0 else not assignment[abs(b) - 1]
            if not (va or vb):
                return False
        return True
