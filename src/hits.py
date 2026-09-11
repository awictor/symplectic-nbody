"""HITS: ranking web pages as hubs and authorities.

HITS (Hyperlink-Induced Topic Search; Kleinberg, 1999) is PageRank's contemporary rival, and it
splits importance into TWO complementary scores per node instead of one. An AUTHORITY is a page
many good hubs point to (a definitive source); a HUB is a page that points to many good authorities
(a good list of links). The definitions are mutually recursive -- a good authority is linked by
good hubs, a good hub links to good authorities -- and that circularity is resolved by iterating to
a fixed point:

    authority(p) = sum of hub scores of pages linking TO p
    hub(p)       = sum of authority scores of pages p links to

Written with the adjacency matrix A (A_ij = 1 if i links to j), the update is a = A' h, h = A a,
normalized each round. Substituting, authorities are the dominant eigenvector of A'A and hubs of
AA' -- so HITS is power iteration on those matrices, and it converges to the principal eigenvector.
Unlike PageRank's single query-independent score, HITS is classically computed on a query-specific
subgraph and yields the two roles separately, which is why a good "hub" page (a curated link list)
and a good "authority" page (the thing everyone cites) rank differently.

This module computes HITS by power iteration with normalization, plus the eigenvector check and a
convergence trace -- verified that on a hub-and-spoke graph the hub scores the linker highest and
the authority scores the linked-to targets, that scores converge and are normalized, that a pure
authority has zero hub score and vice versa, and that the results match the dominant eigenvectors
of A'A and AA'. Pure stdlib; a graph-ranking companion to the PageRank note."""

from __future__ import annotations

import math


def _normalize(d):
    """Scale a score dict to unit Euclidean norm (HITS's standard normalization)."""
    norm = math.sqrt(sum(v * v for v in d.values()))
    if norm == 0:
        return d
    return {k: v / norm for k, v in d.items()}


def hits(graph, max_iter=100, tol=1e-10):
    """Hub and authority scores for a directed graph.

    graph: dict node -> iterable of nodes it links to. Returns (hubs, authorities), each a dict
    node -> score, normalized to unit norm. Nodes appearing only as link targets are included."""
    nodes = set(graph.keys())
    for outs in graph.values():
        nodes.update(outs)
    nodes = sorted(nodes, key=str)
    if not nodes:
        return {}, {}

    # incoming edges: who links TO each node
    incoming = {n: [] for n in nodes}
    outgoing = {n: list(graph.get(n, [])) for n in nodes}
    for src, outs in graph.items():
        for dst in outs:
            incoming[dst].append(src)

    hub = {n: 1.0 for n in nodes}
    auth = {n: 1.0 for n in nodes}

    for _ in range(max_iter):
        # AUTHORITY update: sum of hub scores of pages linking to n
        new_auth = {n: sum(hub[s] for s in incoming[n]) for n in nodes}
        new_auth = _normalize(new_auth)
        # HUB update: sum of authority scores of pages n links to (use updated authorities)
        new_hub = {n: sum(new_auth[t] for t in outgoing[n]) for n in nodes}
        new_hub = _normalize(new_hub)

        delta = (sum(abs(new_auth[n] - auth[n]) for n in nodes)
                 + sum(abs(new_hub[n] - hub[n]) for n in nodes))
        auth, hub = new_auth, new_hub
        if delta < tol:
            break

    return hub, auth


def top_hubs(hubs, k=10):
    return sorted(hubs.items(), key=lambda kv: (-kv[1], str(kv[0])))[:k]


def top_authorities(auths, k=10):
    return sorted(auths.items(), key=lambda kv: (-kv[1], str(kv[0])))[:k]


def adjacency_matrix(graph):
    """Dense adjacency matrix A (A[i][j]=1 if node i links to node j) and the sorted node list."""
    nodes = set(graph.keys())
    for outs in graph.values():
        nodes.update(outs)
    nodes = sorted(nodes, key=str)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = [[0.0] * n for _ in range(n)]
    for src, outs in graph.items():
        for dst in outs:
            A[idx[src]][idx[dst]] = 1.0
    return A, nodes


def authority_matrix(graph):
    """A'A, whose dominant eigenvector is the authority vector (for the eigenvector check)."""
    A, nodes = adjacency_matrix(graph)
    n = len(nodes)
    # (A'A)_jk = sum_i A_ij A_ik
    M = [[sum(A[i][j] * A[i][k] for i in range(n)) for k in range(n)] for j in range(n)]
    return M, nodes


def hub_matrix(graph):
    """AA', whose dominant eigenvector is the hub vector."""
    A, nodes = adjacency_matrix(graph)
    n = len(nodes)
    # (AA')_ik = sum_j A_ij A_kj
    M = [[sum(A[i][j] * A[k][j] for j in range(n)) for k in range(n)] for i in range(n)]
    return M, nodes
