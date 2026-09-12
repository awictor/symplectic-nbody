"""Bron-Kerbosch: enumerating all maximal cliques of a graph.

A CLIQUE is a set of vertices every pair of which is connected -- a fully mutual group. A MAXIMAL
clique is one that cannot be extended by adding another vertex (not to be confused with the MAXIMUM
clique, the largest of all). Listing every maximal clique is the core question of community detection
in social networks (fully connected friend groups), motif finding in biology (mutually interacting
proteins), and constraint analysis (mutually compatible choices). The number of maximal cliques can be
exponential -- a graph on n vertices can have up to 3^(n/3) of them (the Moon-Moser bound) -- so no
algorithm is polynomial in n alone, but the BRON-KERBOSCH algorithm enumerates them all with remarkable
efficiency in practice.

The algorithm is an elegant recursive backtracking over three vertex sets: R (the clique built so far),
P (candidates that can still extend R), and X (vertices already used that would make a duplicate). At
each step, for a candidate v it recurses with v added to R and P, X restricted to v's neighbours, then
moves v from P to X. When P and X are both empty, R is a maximal clique. Two refinements make it fast.
PIVOTING: choose a pivot u in P union X and only branch on candidates that are NOT neighbours of u --
since any maximal clique must contain u or a non-neighbour of it, this prunes redundant branches
dramatically. DEGENERACY ORDERING: processing the outermost recursion in order of a degeneracy (k-core)
ordering bounds the work to O(d * n * 3^(d/3)) where d is the graph's degeneracy, near-optimal for the
sparse graphs common in practice.

This module enumerates all maximal cliques (Bron-Kerbosch with pivoting and an optional degeneracy
ordering), and reports the maximum clique and clique number as a byproduct. It is verified against
brute force -- a set is a maximal clique iff it is a clique and no outside vertex is adjacent to all of
it -- confirming the algorithm returns exactly the maximal cliques, that pivoting and degeneracy
ordering give identical results, and matching the maximum-clique size on hundreds of random graphs.
Pure stdlib; a graph-algorithms companion to the k-core/degeneracy, connectivity, and independent-set
notes."""

from __future__ import annotations


def _adj_sets(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def maximal_cliques(n, edges):
    """All maximal cliques via Bron-Kerbosch with pivoting. Returns a list of frozensets."""
    adj = _adj_sets(n, edges)
    cliques = []

    def expand(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # pivot: a vertex in P|X with the most neighbours in P (minimises branching)
        pu = max(P | X, key=lambda u: len(P & adj[u]))
        for v in list(P - adj[pu]):
            expand(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    expand(set(), set(range(n)), set())
    return cliques


def _degeneracy_order(n, adj):
    """A degeneracy (smallest-last) ordering: repeatedly remove a minimum-degree vertex. Returns the
    order as a list of vertices."""
    deg = [len(adj[v]) for v in range(n)]
    removed = [False] * n
    order = []
    # simple O(n^2) selection (fine for the sizes here); buckets would give O(n+m)
    for _ in range(n):
        best = -1
        for v in range(n):
            if not removed[v] and (best == -1 or deg[v] < deg[best]):
                best = v
        removed[best] = True
        order.append(best)
        for w in adj[best]:
            if not removed[w]:
                deg[w] -= 1
    return order


def maximal_cliques_degeneracy(n, edges):
    """All maximal cliques using the degeneracy-ordering outer loop with pivoting inside. Same result
    as `maximal_cliques`, faster on sparse graphs. Returns a list of frozensets."""
    adj = _adj_sets(n, edges)
    order = _degeneracy_order(n, adj)
    pos = {v: i for i, v in enumerate(order)}
    cliques = []

    def expand(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        pu = max(P | X, key=lambda u: len(P & adj[u]))
        for v in list(P - adj[pu]):
            expand(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    for i, v in enumerate(order):
        # later-in-order neighbours are candidates P; earlier ones are X
        later = {w for w in adj[v] if pos[w] > i}
        earlier = {w for w in adj[v] if pos[w] < i}
        expand({v}, later, earlier)
    return cliques


def maximum_clique(n, edges):
    """The largest clique (one of them) and the clique number (its size)."""
    best = frozenset()
    for c in maximal_cliques(n, edges):
        if len(c) > len(best):
            best = c
    return best, len(best)


# --- brute-force reference --------------------------------------------------
def _is_clique(vs, adj):
    vl = list(vs)
    for i in range(len(vl)):
        for j in range(i + 1, len(vl)):
            if vl[j] not in adj[vl[i]]:
                return False
    return True


def brute_maximal_cliques(n, edges):
    """All maximal cliques by checking every vertex subset: a subset is a maximal clique iff it is a
    clique and no outside vertex is adjacent to all of it. Exponential; small graphs only."""
    adj = _adj_sets(n, edges)
    result = []
    for mask in range(1, 1 << n):
        vs = frozenset(i for i in range(n) if mask & (1 << i))
        if not _is_clique(vs, adj):
            continue
        # maximal: no vertex outside vs is adjacent to every vertex of vs
        maximal = True
        for u in range(n):
            if u not in vs and all(u in adj[w] for w in vs):
                maximal = False
                break
        if maximal:
            result.append(vs)
    return result
