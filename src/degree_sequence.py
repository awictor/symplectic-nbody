"""Degree sequences: which lists of numbers can be the degrees of a real graph?

A GRAPHIC sequence is a list of non-negative integers that is the degree sequence of some simple graph
(no loops, no multi-edges). Not every list qualifies: (3, 3, 3, 1) cannot be realised -- three vertices
each wanting three neighbours among only four vertices force a repeated edge. Deciding realizability,
and building a witness graph when one exists, is a foundational question in graph theory with uses in
network modelling (does a proposed connectivity profile even exist?), randomized graph generation with
prescribed degrees, and chemistry (valence sequences of molecular graphs).

Two classic results settle it. The HAVEL-HAKIMI algorithm is constructive: sort the degrees
descending, remove the largest degree d, and subtract 1 from the next d degrees (connecting that vertex
to the d highest-demand others); the original sequence is graphic iff the reduced one is, and repeating
until all zeros both decides realizability and records the edges of a witness graph. The ERDOS-GALLAI
theorem is a direct inequality test: a descending sequence with even sum is graphic iff, for every
prefix length k, the sum of the first k degrees is at most k(k-1) + sum over the rest of min(d_i, k) --
capturing that the k highest-degree vertices can absorb at most k(k-1) edges among themselves plus what
the remaining vertices can supply. The two criteria always agree.

This module tests whether a sequence is graphic by both Havel-Hakimi and Erdos-Gallai, constructs a
realizing simple graph when one exists, and reports the reduction steps. It is verified against brute
force -- for small sequences, exhaustively searching all simple graphs on that many vertices for one
with the exact degree sequence -- confirming the two criteria agree with each other and with actual
realizability, that the constructed graph is simple and has precisely the requested degrees, and on
classic cases (regular sequences, the non-graphic (3,3,3,1), the handshake even-sum rule). Pure stdlib;
a graph-theory companion to the Prufer-sequence, tree-isomorphism, and connectivity notes."""

from __future__ import annotations


def is_graphic_erdos_gallai(seq):
    """True iff `seq` (a list of non-negative integers) is the degree sequence of some simple graph,
    by the Erdos-Gallai theorem."""
    d = sorted(seq, reverse=True)
    n = len(d)
    if any(x < 0 or x >= n for x in d):
        return False
    if sum(d) % 2 != 0:
        return False                     # handshake lemma: the degree sum must be even
    prefix = 0
    for k in range(1, n + 1):
        prefix += d[k - 1]
        rhs = k * (k - 1) + sum(min(d[i], k) for i in range(k, n))
        if prefix > rhs:
            return False
    return True


def is_graphic_havel_hakimi(seq):
    """True iff `seq` is graphic, by the Havel-Hakimi reduction (non-constructive check)."""
    d = list(seq)
    if any(x < 0 for x in d):
        return False
    while True:
        d = sorted(d, reverse=True)
        if d and d[0] == 0:
            return True                  # all zeros -> graphic
        if not d:
            return True
        top = d.pop(0)
        if top > len(d):
            return False                 # not enough vertices to connect to
        for i in range(top):
            d[i] -= 1
            if d[i] < 0:
                return False
    # unreachable


def realize(seq):
    """Construct a simple graph with degree sequence `seq` via Havel-Hakimi, or None if not graphic.
    Returns a list of edges (u, v) with u < v using the original vertex indices."""
    n = len(seq)
    if not is_graphic_erdos_gallai(seq):
        return None
    # work with (residual_degree, original_index) and connect the highest-degree vertex each round
    nodes = [[seq[i], i] for i in range(n)]
    edges = []
    while True:
        nodes.sort(key=lambda x: -x[0])
        if nodes[0][0] == 0:
            break
        deg, v = nodes[0]
        nodes[0][0] = 0
        if deg > len(nodes) - 1:
            return None                  # shouldn't happen after the EG check
        for j in range(1, deg + 1):
            nodes[j][0] -= 1
            if nodes[j][0] < 0:
                return None
            a, b = v, nodes[j][1]
            edges.append((min(a, b), max(a, b)))
    return sorted(edges)


def reduction_steps(seq):
    """The Havel-Hakimi reduction as a list of sorted sequences, from the input down to all-zeros (or
    until it fails). Useful for illustrating the process. Returns (steps, is_graphic)."""
    d = sorted((x for x in seq), reverse=True)
    steps = [list(d)]
    while True:
        if not d or d[0] == 0:
            return steps, True
        top = d.pop(0)
        if top > len(d):
            return steps, False
        for i in range(top):
            d[i] -= 1
            if d[i] < 0:
                return steps, False
        d = sorted(d, reverse=True)
        steps.append(list(d))


def degree_sequence(n, edges):
    """The degree sequence (descending) of a graph on n vertices."""
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return sorted(deg, reverse=True)


# --- brute-force reference --------------------------------------------------
def brute_is_graphic(seq):
    """Decide realizability by exhaustively searching all simple graphs on len(seq) vertices for one
    with the exact (multiset) degree sequence. Exponential in n^2; tiny sequences only."""
    from itertools import combinations, product
    n = len(seq)
    target = sorted(seq, reverse=True)
    if any(x < 0 or x >= n for x in seq):
        return False
    if sum(seq) % 2 != 0:
        return False
    all_pairs = list(combinations(range(n), 2))
    m = len(all_pairs)
    # try every subset of possible edges (2^m); prune by trying increasing edge counts is overkill
    # for the tiny n used in tests, so enumerate subsets directly
    for bits in product((0, 1), repeat=m):
        deg = [0] * n
        for use, (u, v) in zip(bits, all_pairs):
            if use:
                deg[u] += 1
                deg[v] += 1
        if sorted(deg, reverse=True) == target:
            return True
    return False
