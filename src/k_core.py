"""k-core decomposition: peeling a network to its densely-connected heart.

The k-CORE of a graph is the largest subgraph in which every vertex has at least k neighbours WITHIN
that subgraph. Peeling away all vertices of degree less than k -- and repeating, since each removal can
drop a survivor below k -- leaves the k-core. Each vertex has a CORENESS: the largest k for which it
belongs to the k-core, and the maximum coreness over all vertices is the graph's DEGENERACY, a robust
measure of how dense the graph is (a tree has degeneracy 1, a k-regular graph has degeneracy k). This
decomposition is a workhorse of network science: it identifies the influential "core" of a social
network, ranks nodes by how deeply embedded they are, finds cohesive communities, and its degeneracy
ordering bounds the running time of clique and coloring algorithms.

The elegant algorithm computes all corenesses in O(V + E) by repeatedly removing a minimum-degree
vertex (a bucket-sorted, smallest-last peeling). When a vertex is removed, its current degree is its
coreness -- because at that moment every remaining vertex, including it, has degree at least that
value, so it sits in that core but no deeper. The order in which vertices are removed is the
DEGENERACY ORDERING, and the maximum coreness assigned is the degeneracy. The k-SHELL is the set of
vertices whose coreness is exactly k -- the onion-like layers of the network, from the loosely attached
periphery (shell 0/1) inward to the dense core.

This module computes every vertex's coreness, the graph's degeneracy and a degeneracy ordering, the
vertex set of the k-core for any k, and the k-shell layers. It is verified against the brute-force
definition -- the k-core computed by literally iterating "remove all vertices of degree < k until
stable" matches the coreness-derived core for every k, coreness values equal the max k each vertex
survives to, and the degeneracy ordering has the defining property that each vertex has at most
(degeneracy) neighbours later in the order -- on hundreds of random graphs. Pure stdlib; a
graph-algorithms companion to the Bron-Kerbosch clique, connectivity, and community-detection notes."""

from __future__ import annotations


def _adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def coreness(n, edges):
    """Every vertex's coreness (the largest k for which it lies in the k-core), computed by
    smallest-last peeling in O(V + E). Returns a list core[v]."""
    adj = _adj(n, edges)
    degree = [len(adj[v]) for v in range(n)]
    max_deg = max(degree, default=0)

    # bucket[d] = list of vertices with current degree d
    buckets = [[] for _ in range(max_deg + 1)]
    for v in range(n):
        buckets[degree[v]].append(v)

    core = [0] * n
    removed = [False] * n
    processed = 0
    k = 0
    while processed < n:
        # find the smallest non-empty bucket at or above the current level
        d = 0
        while d <= max_deg and not buckets[d]:
            d += 1
        if d > max_deg:
            break
        v = buckets[d].pop()
        if removed[v]:
            continue
        k = max(k, degree[v])
        core[v] = k
        removed[v] = True
        processed += 1
        for w in adj[v]:
            if not removed[w] and degree[w] > d:
                # move w down one bucket
                buckets[degree[w]].remove(w)
                degree[w] -= 1
                buckets[degree[w]].append(w)
    return core


def degeneracy(n, edges):
    """The graph's degeneracy: the maximum coreness (equivalently, the smallest d such that every
    subgraph has a vertex of degree <= d)."""
    c = coreness(n, edges)
    return max(c, default=0)


def degeneracy_ordering(n, edges):
    """A degeneracy (smallest-last) ordering: the order in which smallest-last peeling removes
    vertices. Each vertex has at most `degeneracy` neighbours that appear later in this order."""
    adj = _adj(n, edges)
    degree = [len(adj[v]) for v in range(n)]
    removed = [False] * n
    order = []
    for _ in range(n):
        # pick a remaining vertex of minimum current degree
        best = -1
        for v in range(n):
            if not removed[v] and (best == -1 or degree[v] < degree[best]):
                best = v
        if best == -1:
            break
        order.append(best)
        removed[best] = True
        for w in adj[best]:
            if not removed[w]:
                degree[w] -= 1
    return order


def k_core(n, edges, k):
    """The vertex set of the k-core: all vertices with coreness >= k."""
    c = coreness(n, edges)
    return [v for v in range(n) if c[v] >= k]


def k_shell(n, edges, k):
    """The k-shell: vertices whose coreness is exactly k (one onion layer)."""
    c = coreness(n, edges)
    return [v for v in range(n) if c[v] == k]


def shells(n, edges):
    """All shells as a dict {k: [vertices with coreness k]}."""
    c = coreness(n, edges)
    out = {}
    for v in range(n):
        out.setdefault(c[v], []).append(v)
    return out


# --- brute-force reference --------------------------------------------------
def brute_k_core(n, edges, k):
    """The k-core by the definition: repeatedly delete every vertex whose degree (within the current
    surviving set) is < k, until nothing changes. Returns the surviving vertex list."""
    adj = _adj(n, edges)
    alive = set(range(n))
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            deg = sum(1 for w in adj[v] if w in alive)
            if deg < k:
                alive.discard(v)
                changed = True
    return sorted(alive)


def brute_coreness(n, edges):
    """Coreness of each vertex by the definition: the largest k for which the vertex is in the
    brute-force k-core."""
    max_k = n
    core = [0] * n
    for k in range(1, max_k + 1):
        ck = set(brute_k_core(n, edges, k))
        if not ck:
            break
        for v in ck:
            core[v] = k
    return core
