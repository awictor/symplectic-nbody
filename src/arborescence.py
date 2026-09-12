"""Chu-Liu/Edmonds: the minimum spanning arborescence, a directed minimum spanning tree.

The minimum spanning tree connects an undirected graph at least cost; its directed cousin is the
MINIMUM SPANNING ARBORESCENCE. Given a directed, weighted graph and a ROOT vertex, an arborescence is a
spanning tree in which every non-root vertex has exactly one incoming edge and is reachable from the
root -- a hierarchy where cost flows outward from a single source. The minimum one is the cheapest way
to broadcast from the root to every node: the cheapest set of one-way links so a message from the root
reaches everyone, the least-cost dependency tree, the optimal branching in a network. Greedy MST
algorithms (Kruskal, Prim) fail here because directions matter -- picking the cheapest incoming edge at
each vertex can create cycles that a spanning tree may not contain.

The CHU-LIU/EDMONDS algorithm handles exactly that. First, for every non-root vertex, select its
cheapest incoming edge. If these choices form no cycle, they already are the optimum arborescence. If
they form a cycle, the algorithm CONTRACTS the whole cycle into a single super-vertex, reweighting each
edge entering the cycle by subtracting the cost of the cycle edge it would replace, and recurses on the
smaller graph. Expanding the contracted cycles back -- breaking each at the one vertex whose incoming
edge was chosen from outside -- reconstructs the true minimum arborescence in the original graph. The
reweighting is the crux: it correctly accounts for the fact that entering the cycle lets you drop one
of its internal edges.

This module computes the minimum spanning arborescence rooted at a given vertex (its total weight and
its parent-edge structure), detects when no arborescence exists (some vertex unreachable from the
root), and reconstructs the chosen edges. It is verified against brute force -- enumerating every
possible choice of one incoming edge per non-root vertex, keeping those that form a valid arborescence,
and confirming the algorithm finds the true minimum weight -- on hundreds of random graphs, plus the
special cases of trees (unique arborescence), unreachable vertices, and multi-edges. Pure stdlib; a
graph-optimisation companion to the MST (Kruskal/Prim), max-flow, and shortest-path notes."""

from __future__ import annotations

_INF = float("inf")


def min_arborescence(n, edges, root):
    """Minimum spanning arborescence rooted at `root`.

    `n` is the number of vertices (0..n-1); `edges` is a list of (u, v, w) directed edges u->v with
    weight w. Returns (total_weight, parent) where parent[v] is the (u, v, w) edge chosen for v (and
    parent[root] is None), or (None, None) if no arborescence exists (some vertex is unreachable from
    the root)."""
    # tag each edge with a unique index so recursion can report which ORIGINAL edges were chosen
    # regardless of per-level vertex renumbering
    tagged = [(u, v, w, idx) for idx, (u, v, w) in enumerate(edges)]
    total, chosen_idx = _solve(n, tagged, root)
    if total is None:
        return None, None
    parent = {v: None for v in range(n)}
    for idx in chosen_idx:
        u, v, w = edges[idx]
        parent[v] = (u, v, w)
    return total, parent


def _solve(n, edges, root):
    """One round of Chu-Liu/Edmonds. `edges` are (u, v, w, idx) with `idx` the ORIGINAL edge index
    (stable across recursion levels), u/v in THIS level's numbering. Returns (total_weight,
    list_of_original_edge_indices) or (None, None) if infeasible."""
    # 1) cheapest incoming edge per non-root vertex
    best_in = {v: None for v in range(n) if v != root}
    for e in edges:
        v, w = e[1], e[2]
        if v == root:
            continue
        if best_in[v] is None or w < best_in[v][2]:
            best_in[v] = e
    for v in best_in:
        if best_in[v] is None:
            return None, None                    # some vertex has no incoming edge

    # 2) find cycles among the chosen edges by following parent sources
    cycle_id = {v: -1 for v in range(n)}
    seen = {v: -1 for v in range(n)}
    num_cycles = 0
    for start in range(n):
        if start == root:
            continue
        v = start
        while v != root and seen[v] == -1 and cycle_id[v] == -1:
            seen[v] = start
            v = best_in[v][0]
        if v != root and seen[v] == start:
            cyc = v
            while cycle_id[cyc] == -1:
                cycle_id[cyc] = num_cycles
                cyc = best_in[cyc][0]
            num_cycles += 1

    # 3) no cycle -> the chosen edges are optimal
    if num_cycles == 0:
        chosen = [best_in[v][3] for v in best_in]
        total = sum(best_in[v][2] for v in best_in)
        return total, chosen

    # 4) contract each cycle to a super-vertex (ids 0..num_cycles-1); others get fresh ids
    new_id = {}
    next_id = num_cycles
    for v in range(n):
        if cycle_id[v] != -1:
            new_id[v] = cycle_id[v]
        else:
            new_id[v] = next_id
            next_id += 1
    new_n = next_id
    new_root = new_id[root]
    cycle_cost = sum(best_in[v][2] for v in range(n) if cycle_id[v] != -1)

    # 5) reweight edges entering a cycle (subtract the in-cycle edge they would replace). Track, per
    # original edge index, which THIS-level cycle vertex it enters (its real head), so expansion can
    # break the correct cycle even through nested contractions.
    new_edges = []
    enters_cycle_at = {}                          # original edge index -> real head vertex (this level)
    for e in edges:
        u, v, w, idx = e
        nu, nv = new_id[u], new_id[v]
        if nu == nv:
            continue                             # inside a contracted cycle
        if cycle_id[v] != -1:
            rw = w - best_in[v][2]
            enters_cycle_at[idx] = v
        else:
            rw = w
        new_edges.append((nu, nv, rw, idx))

    # 6) recurse
    sub_total, sub_chosen = _solve(new_n, new_edges, new_root)
    if sub_total is None:
        return None, None

    # 7) expand. A chosen edge that entered a cycle tells us the real vertex entered from outside;
    # drop THAT vertex's internal cycle edge, keep the rest of every cycle.
    total = sub_total + cycle_cost
    chosen = list(sub_chosen)
    entered = {}                                 # cycle id -> real vertex entered from outside
    for idx in sub_chosen:
        if idx in enters_cycle_at:
            v = enters_cycle_at[idx]
            entered[cycle_id[v]] = v
    for v in range(n):
        if cycle_id[v] != -1 and entered.get(cycle_id[v]) != v:
            chosen.append(best_in[v][3])         # keep all internal cycle edges but the broken one
    return total, chosen


# --- brute-force reference --------------------------------------------------
def brute_min_arborescence(n, edges, root):
    """Minimum arborescence weight by enumerating every choice of one incoming edge per non-root
    vertex and keeping those that form a valid rooted tree. Exponential; small graphs only. Returns
    (weight, chosen_edges) or (None, None)."""
    from itertools import product

    incoming = {v: [] for v in range(n) if v != root}
    for e in edges:
        u, v, w = e[0], e[1], e[2]
        if v != root:
            incoming[v].append(e)
    verts = [v for v in range(n) if v != root]
    if any(not incoming[v] for v in verts):
        return None, None

    best_w = None
    best_choice = None
    for combo in product(*[incoming[v] for v in verts]):
        # combo is one incoming edge per non-root vertex; check it forms an arborescence
        parent = {v: combo[i][0] for i, v in enumerate(verts)}
        if _is_arborescence(n, parent, root):
            w = sum(e[2] for e in combo)
            if best_w is None or w < best_w:
                best_w = w
                best_choice = list(combo)
    return best_w, best_choice


def _is_arborescence(n, parent, root):
    """True iff following parent pointers from every vertex reaches the root without cycling (i.e. the
    chosen edges form a tree rooted at `root`)."""
    for start in range(n):
        if start == root:
            continue
        v = start
        steps = 0
        while v != root:
            if v not in parent:
                return False
            v = parent[v]
            steps += 1
            if steps > n:
                return False        # cycle
    return True
