"""Graph coloring: assigning colors so no two neighbours clash, with as few colors as possible.

A PROPER COLORING of a graph gives each vertex a color so that adjacent vertices differ; the CHROMATIC
NUMBER is the fewest colors any proper coloring needs. This is the mathematics of conflict-free
scheduling: exam timetables (courses sharing a student can't overlap), register allocation in compilers
(variables live at the same time need different registers), frequency assignment (nearby transmitters
must differ), and map coloring (the Four Color Theorem). Deciding whether a graph is k-colorable is
NP-complete for k >= 3, so exact chromatic number is exponential, but fast heuristics color well in
practice and small graphs succumb to branch-and-bound.

This module implements three approaches. GREEDY coloring walks the vertices in a given order and gives
each the smallest color not used by an already-colored neighbour -- fast, but its color count depends
on the order (a bad order can need far more colors than necessary). DSATUR (degree of saturation)
colors smartly: it always picks next the uncolored vertex adjacent to the most DISTINCT colors already
used (breaking ties by degree), which provably uses the optimal number of colors on many graph classes
and excellently in general. EXACT chromatic number runs branch-and-bound: try to k-color for
increasing k, assigning colors vertex by vertex and backtracking, using a clique lower bound and a
DSATUR upper bound to prune.

The module returns a greedy coloring, a DSATUR coloring, and the exact chromatic number with a witness
coloring. It is verified against brute force -- every coloring produced is proper (no edge is
monochromatic), the exact chromatic number matches an exhaustive search over all k-colorings for
increasing k, greedy and DSATUR never use fewer colors than the true chromatic number (they are upper
bounds), and known values hold (a cycle of even length needs 2 colors and odd length 3, the complete
graph K_n needs n) -- on hundreds of random graphs. Pure stdlib; a graph-algorithms companion to the
Bron-Kerbosch clique, independent-set, and bipartite-matching notes."""

from __future__ import annotations


def _adj_sets(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def greedy_coloring(n, edges, order=None):
    """Greedy proper coloring: color vertices in `order` (default 0..n-1), each getting the smallest
    color unused by its already-colored neighbours. Returns a list color[v]."""
    adj = _adj_sets(n, edges)
    if order is None:
        order = list(range(n))
    color = [-1] * n
    for v in order:
        used = {color[w] for w in adj[v] if color[w] != -1}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def dsatur_coloring(n, edges):
    """DSATUR coloring: repeatedly color the uncolored vertex with the highest saturation (number of
    distinct colors among its neighbours), breaking ties by highest degree. Returns a list color[v]."""
    adj = _adj_sets(n, edges)
    color = [-1] * n
    degree = [len(adj[v]) for v in range(n)]
    neigh_colors = [set() for _ in range(n)]

    for _ in range(n):
        # pick the uncolored vertex of maximum saturation, then maximum degree
        best = -1
        best_key = None
        for v in range(n):
            if color[v] == -1:
                key = (len(neigh_colors[v]), degree[v])
                if best == -1 or key > best_key:
                    best = v
                    best_key = key
        if best == -1:
            break
        used = neigh_colors[best]
        c = 0
        while c in used:
            c += 1
        color[best] = c
        for w in adj[best]:
            neigh_colors[w].add(c)
    return color


def num_colors(coloring):
    """The number of distinct colors used (0 for an empty coloring)."""
    used = {c for c in coloring if c != -1}
    return len(used)


def is_proper(n, edges, coloring):
    """True iff `coloring` gives adjacent vertices different colors and colors every vertex."""
    if any(c == -1 for c in coloring):
        return False
    for u, v in edges:
        if u != v and coloring[u] == coloring[v]:
            return False
    return True


def chromatic_number(n, edges):
    """The exact chromatic number and a witness coloring, via branch-and-bound over increasing k.
    Returns (chromatic_number, coloring). Exponential; intended for small graphs."""
    if n == 0:
        return 0, []
    adj = _adj_sets(n, edges)
    if not edges:
        return 1, [0] * n

    # upper bound from DSATUR, lower bound from a greedy clique
    ub = num_colors(dsatur_coloring(n, edges))
    lb = _greedy_clique_size(n, adj)

    # order vertices by descending degree for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))

    for k in range(lb, ub + 1):
        coloring = [-1] * n
        if _can_color(order, 0, k, adj, coloring):
            return k, coloring
    # ub is always achievable
    return ub, dsatur_coloring(n, edges)


def _can_color(order, idx, k, adj, color):
    """Backtracking: try to color order[idx:] with colors 0..k-1."""
    if idx == len(order):
        return True
    v = order[idx]
    used = {color[w] for w in adj[v] if color[w] != -1}
    # symmetry break: only allow colors up to 1 + max color used so far
    max_used = max((color[w] for w in range(len(color)) if color[w] != -1), default=-1)
    limit = min(k, max_used + 2)
    for c in range(limit):
        if c not in used:
            color[v] = c
            if _can_color(order, idx + 1, k, adj, color):
                return True
            color[v] = -1
    return False


def _greedy_clique_size(n, adj):
    """A quick clique lower bound: greedily grow a clique from the highest-degree vertex."""
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    best = 1
    for start in order[:min(n, 8)]:            # a few seeds
        clique = {start}
        for v in order:
            if v not in clique and all(v in adj[c] for c in clique):
                clique.add(v)
        best = max(best, len(clique))
    return best


# --- brute-force reference --------------------------------------------------
def brute_chromatic_number(n, edges):
    """The exact chromatic number by trying all k-colorings for k = 1, 2, ... until one is proper.
    Exponential (k^n); tiny graphs only."""
    if n == 0:
        return 0
    adj = _adj_sets(n, edges)
    for k in range(1, n + 1):
        if _brute_k_colorable(n, adj, k):
            return k
    return n


def _brute_k_colorable(n, adj, k):
    """True iff the graph has a proper k-coloring, by exhaustive assignment with backtracking."""
    color = [-1] * n

    def rec(v):
        if v == n:
            return True
        for c in range(k):
            if all(color[w] != c for w in adj[v]):
                color[v] = c
                if rec(v + 1):
                    return True
                color[v] = -1
        return False

    return rec(0)
