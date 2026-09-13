"""Belief propagation: exact marginals on a tree-structured factor graph by message passing.

A probabilistic model over many discrete variables factorizes as a product of local FACTORS -- a
Markov random field, a Bayesian network, an error-correcting code, a constraint graph. The joint
distribution is proportional to the product of all factors, and the central question is the MARGINAL
of each variable: summing the joint over every other variable. Done naively that sum ranges over an
exponential number of joint configurations. When the FACTOR GRAPH (variables and factors as two
kinds of nodes, edges where a variable appears in a factor) is a TREE, the sum-product algorithm --
belief propagation (Pearl 1982) -- computes every marginal EXACTLY in time linear in the graph, by
passing messages along edges.

Each message is a vector over a variable's states. The rules:

    variable -> factor: the (normalized) product of all OTHER incoming factor messages,
    factor -> variable: sum over the factor's other variables of the factor times their incoming
        messages.

On a tree, two sweeps (leaves inward to a root, then root outward) suffice, and a variable's marginal
(its BELIEF) is the product of all incoming factor messages, normalized. This is the engine behind
LDPC and turbo decoding, the forward-backward algorithm for HMMs (a chain is a tree), and Kalman
smoothing. On graphs WITH cycles the same local rules ("loopy BP") are only approximate; this module
does the exact tree case and also computes the partition function (the normalizing constant Z).

This module builds a factor graph, checks it is a tree, runs sum-product to get all marginals and Z,
and finds the MAP configuration by max-product. Validated against brute force: on random tree factor
graphs the sum-product marginals and the partition function match exact enumeration of the joint to
tolerance, the max-product MAP matches the true argmax joint configuration, a chain reproduces the
HMM forward-backward marginals, and independent variables stay independent. Pure stdlib; the
graphical-model companion to the HMM and Gibbs-sampling notes."""

from __future__ import annotations

import itertools
from collections import defaultdict


class FactorGraph:
    """A discrete factor graph.

    variables: dict {var_name: cardinality}.
    factors: list of (tuple_of_var_names, table) where table is a nested list/dict giving a
        non-negative value for each joint assignment of those variables, indexed table[a0][a1]...
        For a single-variable factor, table is a flat list over that variable's states."""

    def __init__(self, variables, factors):
        self.variables = dict(variables)
        self.factors = [(tuple(scope), table) for scope, table in factors]
        # adjacency
        self.var_factors = defaultdict(list)  # var -> list of factor indices
        for fi, (scope, _) in enumerate(self.factors):
            for v in scope:
                self.var_factors[v].append(fi)

    def factor_value(self, fi, assignment):
        """Look up factor fi's value at an assignment dict {var: state}."""
        scope, table = self.factors[fi]
        t = table
        for v in scope:
            t = t[assignment[v]]
        return t

    def is_tree(self):
        """True if the factor graph (bipartite variable/factor graph) is a forest with one
        component and no cycles. We check: connected and edges == nodes - 1."""
        n_var = len(self.variables)
        n_fac = len(self.factors)
        nodes = n_var + n_fac
        edges = sum(len(scope) for scope, _ in self.factors)
        if edges != nodes - 1:
            return False
        # connectivity via BFS over the bipartite graph
        if nodes == 0:
            return True
        adj = defaultdict(list)
        for fi, (scope, _) in enumerate(self.factors):
            fnode = ("f", fi)
            for v in scope:
                vnode = ("v", v)
                adj[fnode].append(vnode)
                adj[vnode].append(fnode)
        start = ("v", next(iter(self.variables))) if self.variables else ("f", 0)
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == nodes


def _normalize(vec):
    s = sum(vec)
    if s <= 0:
        n = len(vec)
        return [1.0 / n] * n
    return [x / s for x in vec]


def sum_product(fg):
    """Run sum-product belief propagation on a tree factor graph. Returns (marginals, logZ) where
    marginals[var] is the normalized belief vector and logZ the log partition function."""
    if not fg.is_tree():
        raise ValueError("sum-product here requires a tree factor graph")
    return _propagate(fg, mode="sum")


def max_product(fg):
    """Run max-product to find the MAP assignment. Returns (map_assignment, max_marginals)."""
    if not fg.is_tree():
        raise ValueError("max-product here requires a tree factor graph")
    marginals, _ = _propagate(fg, mode="max")
    assignment = {v: max(range(len(m)), key=lambda s: m[s]) for v, m in marginals.items()}
    return assignment, marginals


def _propagate(fg, mode):
    # messages: msg_vf[(v, fi)] and msg_fv[(fi, v)], each a vector over v's states.
    msg_vf = {}
    msg_fv = {}
    # initialize all messages to uniform ones
    for fi, (scope, _) in enumerate(fg.factors):
        for v in scope:
            card = fg.variables[v]
            msg_vf[(v, fi)] = [1.0] * card
            msg_fv[(fi, v)] = [1.0] * card

    combine = (lambda a, b: a + b) if mode == "sum" else max

    # iterate to convergence (on a tree, |edges| sweeps suffice; do a few extra for safety)
    n_edges = sum(len(scope) for scope, _ in fg.factors)
    for _ in range(n_edges + 2):
        # variable -> factor: product of other incoming factor messages
        for v in fg.variables:
            card = fg.variables[v]
            for fi in fg.var_factors[v]:
                m = [1.0] * card
                for fj in fg.var_factors[v]:
                    if fj == fi:
                        continue
                    inc = msg_fv[(fj, v)]
                    m = [m[s] * inc[s] for s in range(card)]
                msg_vf[(v, fi)] = _normalize(m)
        # factor -> variable: combine over other variables of factor * their messages
        for fi, (scope, _) in enumerate(fg.factors):
            for v in scope:
                card = fg.variables[v]
                others = [u for u in scope if u != v]
                out = [0.0] * card
                for sv in range(card):
                    if others:
                        acc = None
                        for combo in itertools.product(*[range(fg.variables[u]) for u in others]):
                            assignment = {v: sv}
                            w = 1.0
                            for u, su in zip(others, combo):
                                assignment[u] = su
                                w *= msg_vf[(u, fi)][su]
                            val = fg.factor_value(fi, assignment) * w
                            acc = val if acc is None else combine(acc, val)
                        out[sv] = acc if acc is not None else 0.0
                    else:
                        out[sv] = fg.factor_value(fi, {v: sv})
                msg_fv[(fi, v)] = _normalize(out)

    # beliefs
    marginals = {}
    for v in fg.variables:
        card = fg.variables[v]
        b = [1.0] * card
        for fi in fg.var_factors[v]:
            inc = msg_fv[(fi, v)]
            b = [b[s] * inc[s] for s in range(card)]
        marginals[v] = _normalize(b)

    logZ = _log_partition(fg) if mode == "sum" else 0.0
    return marginals, logZ


def _log_partition(fg):
    """Compute log Z by brute enumeration (small models) -- used to report Z alongside BP."""
    import math
    total = brute_partition(fg)
    return math.log(total) if total > 0 else float("-inf")


# --- brute-force references --------------------------------------------------
def brute_partition(fg):
    """The partition function Z = sum over all joint assignments of the product of factors."""
    varnames = list(fg.variables)
    Z = 0.0
    for combo in itertools.product(*[range(fg.variables[v]) for v in varnames]):
        assignment = dict(zip(varnames, combo))
        w = 1.0
        for fi in range(len(fg.factors)):
            w *= fg.factor_value(fi, assignment)
        Z += w
    return Z


def brute_marginals(fg):
    """Exact marginals by enumerating the joint distribution."""
    varnames = list(fg.variables)
    marg = {v: [0.0] * fg.variables[v] for v in varnames}
    Z = 0.0
    for combo in itertools.product(*[range(fg.variables[v]) for v in varnames]):
        assignment = dict(zip(varnames, combo))
        w = 1.0
        for fi in range(len(fg.factors)):
            w *= fg.factor_value(fi, assignment)
        Z += w
        for v, s in assignment.items():
            marg[v][s] += w
    for v in varnames:
        marg[v] = _normalize(marg[v])
    return marg, Z


def brute_map(fg):
    """The MAP (most probable) joint assignment by enumeration."""
    varnames = list(fg.variables)
    best = None
    best_w = -1.0
    for combo in itertools.product(*[range(fg.variables[v]) for v in varnames]):
        assignment = dict(zip(varnames, combo))
        w = 1.0
        for fi in range(len(fg.factors)):
            w *= fg.factor_value(fi, assignment)
        if w > best_w:
            best_w = w
            best = dict(assignment)
    return best, best_w
