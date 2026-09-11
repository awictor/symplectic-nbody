"""PageRank: ranking a graph by the stationary distribution of a random walk.

PageRank (Brin & Page, 1998 -- the algorithm that launched Google) scores every node of a directed
graph by one idea: a node is important if important nodes link to it. Imagine a random surfer who,
at each step, with probability d follows a random out-link of the current page and with probability
1 - d teleports to a page chosen uniformly at random. PageRank is the fraction of time the surfer
spends on each page in the long run -- the STATIONARY DISTRIBUTION of that Markov chain.

Formally it is the dominant eigenvector of the Google matrix G = d M + (1 - d)/N * 11', where M is
the column-stochastic link matrix, d ~ 0.85 is the damping factor, and the rank-one term is the
teleport. The teleport makes G strictly positive, so Perron-Frobenius guarantees a unique positive
stationary vector and the POWER ITERATION -- repeatedly applying G to any start distribution --
converges to it geometrically at rate d. Two subtleties matter: DANGLING nodes (no out-links) would
leak probability, so their mass is redistributed by teleport; and the whole thing is computed
sparsely, never forming the dense NxN matrix.

This module builds PageRank by sparse power iteration with damping and correct dangling-node
handling, plus the personalized variant (a non-uniform teleport vector) -- verified against the
analytic stationary distribution of small chains, against symmetry on a ring, and by checking the
result is a true fixed point of the Google-matrix map that sums to one. Pure stdlib; a Markov-chain
companion to the graph-algorithm and eigenvector notes."""

from __future__ import annotations


def pagerank(graph, damping=0.85, tol=1e-10, max_iter=1000, personalization=None):
    """PageRank scores for a directed graph.

    graph: dict node -> iterable of out-neighbours (nodes with no key are still ranked if they
           appear as a target). Returns dict node -> score, summing to 1.

    damping: probability the surfer follows a link (vs teleporting), classically 0.85.
    personalization: optional dict node -> weight for a non-uniform teleport distribution
                     (defaults to uniform). Weights are normalized internally.
    """
    # collect the full node set (sources and targets)
    nodes = set(graph.keys())
    for outs in graph.values():
        nodes.update(outs)
    nodes = sorted(nodes, key=str)
    n = len(nodes)
    if n == 0:
        return {}
    idx = {node: i for i, node in enumerate(nodes)}

    # out-degree and adjacency in index space
    out = [[] for _ in range(n)]
    for src, outs in graph.items():
        for dst in outs:
            out[idx[src]].append(idx[dst])
    out_deg = [len(o) for o in out]

    # teleport vector
    if personalization is None:
        teleport = [1.0 / n] * n
    else:
        w = [max(0.0, personalization.get(node, 0.0)) for node in nodes]
        s = sum(w)
        teleport = [wi / s for wi in w] if s > 0 else [1.0 / n] * n

    rank = [1.0 / n] * n
    for _ in range(max_iter):
        new = [0.0] * n
        dangling_mass = 0.0
        for i in range(n):
            if out_deg[i] == 0:
                dangling_mass += rank[i]           # will be redistributed by teleport
            else:
                share = rank[i] / out_deg[i]
                for j in out[i]:
                    new[j] += share
        # apply damping, teleport, and dangling redistribution
        leaked = damping * dangling_mass
        for j in range(n):
            new[j] = damping * new[j] + (1.0 - damping) * teleport[j] + leaked * teleport[j]
        # convergence in L1
        err = sum(abs(new[i] - rank[i]) for i in range(n))
        rank = new
        if err < tol:
            break

    # normalize (guards tiny drift) and map back to node keys
    total = sum(rank)
    return {nodes[i]: rank[i] / total for i in range(n)}


def google_matrix_apply(graph, rank, damping=0.85, personalization=None):
    """Apply one step of the Google-matrix map to a rank dict (for fixed-point verification)."""
    nodes = set(graph.keys())
    for outs in graph.values():
        nodes.update(outs)
    nodes = sorted(nodes, key=lambda x: str(x))
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    out = [[] for _ in range(n)]
    for src, outs in graph.items():
        for dst in outs:
            out[idx[src]].append(idx[dst])
    out_deg = [len(o) for o in out]

    if personalization is None:
        teleport = [1.0 / n] * n
    else:
        w = [max(0.0, personalization.get(node, 0.0)) for node in nodes]
        s = sum(w)
        teleport = [wi / s for wi in w] if s > 0 else [1.0 / n] * n

    r = [rank[nodes[i]] for i in range(n)]
    new = [0.0] * n
    dangling = 0.0
    for i in range(n):
        if out_deg[i] == 0:
            dangling += r[i]
        else:
            share = r[i] / out_deg[i]
            for j in out[i]:
                new[j] += share
    leaked = damping * dangling
    for j in range(n):
        new[j] = damping * new[j] + (1.0 - damping) * teleport[j] + leaked * teleport[j]
    return {nodes[i]: new[i] for i in range(n)}


def top_k(scores, k=10):
    """The k highest-scoring nodes as (node, score) pairs, ties broken by node key."""
    return sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))[:k]
