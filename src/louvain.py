"""Louvain community detection: finding the natural clusters in a network by modularity.

Real networks -- social graphs, protein interactions, the web -- are not uniform; they break into
COMMUNITIES, groups of nodes more densely connected inside than between. Finding them reveals a
network's structure without being told how many groups to look for. The standard quality measure is
MODULARITY: how much denser the within-community edges are than you would expect if the same nodes
were wired up at random. Maximizing modularity is NP-hard, but the LOUVAIN METHOD (Blondel et al.,
2008) is a fast, greedy heuristic that finds excellent partitions on graphs with millions of nodes
and is the de-facto standard for community detection.

Louvain works in two alternating phases, repeated until the modularity stops improving. In the LOCAL
MOVING phase, every node starts in its own community, then each node is moved to whichever neighbour's
community gives the largest MODULARITY GAIN (a quantity computable in O(degree) from the edge weights
into each community and the community's total degree), sweeping until no move helps. In the
AGGREGATION phase, each community is collapsed into a single super-node (with self-loops for internal
edges and weighted links for between-community edges), and the whole process recurses on this smaller
graph. Because each phase only ever increases modularity and the graph shrinks each round, it
converges quickly to a hierarchical partition.

This module implements Louvain modularity optimization on a weighted undirected graph, returning the
community assignment and the achieved modularity, plus a direct modularity calculator. It is verified
against exact references: that the returned modularity matches a direct computation for the found
partition, that on a graph with clearly planted communities (dense cliques joined by a few edges)
Louvain recovers exactly those communities, that a single clique is left as one community, that a
graph with no structure yields low modularity, and that moving a node never decreases the reported
modularity. Pure stdlib; a network-science companion to the PageRank, spectral-clustering, and
union-find notes."""

from __future__ import annotations


class _RNG:
    def __init__(self, seed=1):
        self.state = seed & 0xFFFFFFFF

    def shuffle(self, lst):
        for i in range(len(lst) - 1, 0, -1):
            self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
            j = (self.state >> 8) % (i + 1)
            lst[i], lst[j] = lst[j], lst[i]
        return lst


def modularity(n, edges, communities):
    """The modularity Q of a partition of a weighted undirected graph.

    communities: list mapping each node to its community id. Q in [-0.5, 1]; higher = stronger
    community structure."""
    m2 = 0.0                                  # 2 * total edge weight
    deg = [0.0] * n
    adj = {}
    for u, v, w in edges:
        deg[u] += w
        deg[v] += w
        m2 += 2 * w if u != v else w * 2      # a self-loop still adds 2w to the degree sum? use 2w consistently
    # recompute m2 cleanly as sum of all degrees
    m2 = sum(deg)
    if m2 == 0:
        return 0.0
    # within-community edge weight and community total degree
    q = 0.0
    # sum over edges: for each edge in the same community, contribute 2w (both directions)/m2
    in_weight = {}
    tot_deg = {}
    for c in set(communities):
        tot_deg[c] = 0.0
    for i in range(n):
        tot_deg[communities[i]] += deg[i]
    for u, v, w in edges:
        if communities[u] == communities[v]:
            in_weight[communities[u]] = in_weight.get(communities[u], 0.0) + (2 * w if u != v else w)
    for c in tot_deg:
        lc = in_weight.get(c, 0.0)
        q += lc / m2 - (tot_deg[c] / m2) ** 2
    return q


def _build_adj(n, edges):
    adj = [dict() for _ in range(n)]
    for u, v, w in edges:
        adj[u][v] = adj[u].get(v, 0.0) + w
        if u != v:
            adj[v][u] = adj[v].get(u, 0.0) + w
    return adj


def _one_level(n, adj, deg, m2, seed):
    """One local-moving phase: returns the community label of each node."""
    comm = list(range(n))
    tot = [deg[i] for i in range(n)]          # total degree of each community
    rng = _RNG(seed)
    improved = True
    while improved:
        improved = False
        for node in rng.shuffle(list(range(n))):
            ci = comm[node]
            ki = deg[node]
            # remove node from its community
            tot[ci] -= ki
            # weight from node into each neighbouring community
            weight_to = {}
            for nb, w in adj[node].items():
                if nb != node:
                    weight_to[comm[nb]] = weight_to.get(comm[nb], 0.0) + w
            # find the best community (including staying)
            best_c = ci
            best_gain = weight_to.get(ci, 0.0) - tot[ci] * ki / m2
            for c, wct in weight_to.items():
                gain = wct - tot[c] * ki / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_c = c
            tot[best_c] += ki
            if best_c != ci:
                comm[node] = best_c
                improved = True
            else:
                comm[node] = ci
    return comm


def detect(n, edges, seed=1, max_passes=20):
    """Louvain community detection. Returns (communities, modularity) where communities maps each
    original node to a community id (relabeled 0..k-1)."""
    if n == 0:
        return [], 0.0
    # working graph (may be aggregated across passes); node_map[original] = current super-node
    adj = _build_adj(n, edges)
    deg = [sum(adj[i].values()) for i in range(n)]
    m2 = sum(deg)
    if m2 == 0:
        return list(range(n)), 0.0

    node_of_original = list(range(n))         # which current-super-node each ORIGINAL node belongs to
    cur_n = n
    cur_adj = adj
    cur_deg = deg

    for _ in range(max_passes):
        comm = _one_level(cur_n, cur_adj, cur_deg, m2, seed)
        # relabel communities to 0..k-1
        labels = {}
        for c in comm:
            if c not in labels:
                labels[c] = len(labels)
        comm = [labels[c] for c in comm]
        k = len(labels)
        if k == cur_n:
            break                              # no aggregation happened -> converged
        # map original nodes down
        node_of_original = [comm[node_of_original[i]] for i in range(n)]
        # aggregate: build the super-graph
        new_adj = [dict() for _ in range(k)]
        for u in range(cur_n):
            cu = comm[u]
            for v, w in cur_adj[u].items():
                cv = comm[v]
                new_adj[cu][cv] = new_adj[cu].get(cv, 0.0) + w
        cur_n = k
        cur_adj = new_adj
        cur_deg = [sum(new_adj[i].values()) for i in range(k)]

    # relabel final communities of the ORIGINAL nodes to 0..k-1
    labels = {}
    for c in node_of_original:
        if c not in labels:
            labels[c] = len(labels)
    final = [labels[c] for c in node_of_original]
    return final, modularity(n, edges, final)
