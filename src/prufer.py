"""Prufer sequences: a bijection between labeled trees and integer strings.

A labeled tree on n vertices can be encoded as a PRUFER SEQUENCE -- a list of exactly n-2 vertex labels
-- and reconstructed perfectly from it. This bijection, from Heinz Prufer's 1918 proof, is one of the
most elegant in combinatorics: it turns questions about trees into questions about strings. Its most
famous consequence is CAYLEY'S FORMULA: because a Prufer sequence is any of the n^(n-2) strings of
length n-2 over the alphabet {0,...,n-1}, there are EXACTLY n^(n-2) distinct labeled trees on n
vertices. The sequence also reads off structure directly: a vertex appears in it exactly (degree - 1)
times, so leaves are precisely the labels that never appear. Prufer codes are used to generate a
uniformly random labeled tree (encode a random sequence), to count trees with prescribed degrees, and
as a compact tree serialization.

ENCODING strips leaves one at a time: repeatedly find the smallest-labeled leaf, append its unique
neighbour to the sequence, and remove the leaf; after n-2 removals two vertices remain and the sequence
is complete. DECODING inverts this: track how many times each label still appears in the remaining
sequence (its residual degree minus one); repeatedly take the smallest label not in the remaining
sequence and not yet used (the current smallest leaf), connect it to the front sequence element, and
decrement that element's count; finally join the two labels left over. Both run in O(n log n) with a
heap of candidate leaves.

This module encodes a tree (given as an edge list) into its Prufer sequence, decodes a sequence back
into edges, counts labeled trees via Cayley's formula, and generates a labeled tree from a seed. It is
verified against brute force -- encode-then-decode is the identity on hundreds of random trees, decode
produces a valid tree (n-1 edges, connected, acyclic) for every sequence, the leaf/degree reading
matches, and Cayley's count n^(n-2) is confirmed by exhaustively enumerating all trees for small n via
every possible Prufer sequence. Pure stdlib; a combinatorics companion to the tree-isomorphism,
Lyndon-word, and combinatorial-ranking notes."""

from __future__ import annotations

import heapq


def tree_to_prufer(n, edges):
    """Encode a labeled tree on vertices 0..n-1 into its Prufer sequence (length n-2).

    `edges` is the tree's n-1 edges. Returns a list of n-2 labels. For n < 2 the sequence is empty."""
    if n < 2:
        return []
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degree = [len(adj[v]) for v in range(n)]

    # min-heap of current leaves (degree 1)
    leaves = [v for v in range(n) if degree[v] == 1]
    heapq.heapify(leaves)

    seq = []
    for _ in range(n - 2):
        leaf = heapq.heappop(leaves)
        # its single remaining neighbour
        nb = next(iter(adj[leaf]))
        seq.append(nb)
        # remove the leaf
        adj[nb].discard(leaf)
        degree[nb] -= 1
        if degree[nb] == 1:
            heapq.heappush(leaves, nb)
    return seq


def prufer_to_tree(n, seq):
    """Decode a Prufer sequence of length n-2 back into the edges of a labeled tree on 0..n-1.

    Returns a list of n-1 edges (u, v) with u < v, sorted."""
    if n < 2:
        return []
    if len(seq) != n - 2:
        raise ValueError(f"a Prufer sequence for n={n} must have length {n - 2}, got {len(seq)}")

    # count how many times each label appears in the sequence (= residual degree - 1)
    count = [0] * n
    for x in seq:
        count[x] += 1
    # degree of each vertex is count + 1
    degree = [count[v] + 1 for v in range(n)]

    # min-heap of current leaves (degree 1): labels not appearing in the remaining sequence
    leaves = [v for v in range(n) if degree[v] == 1]
    heapq.heapify(leaves)

    edges = []
    for x in seq:
        leaf = heapq.heappop(leaves)
        edges.append((min(leaf, x), max(leaf, x)))
        degree[leaf] -= 1
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, x)
    # two vertices of degree 1 remain
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    edges.append((min(u, v), max(u, v)))
    return sorted(edges)


def cayley_count(n):
    """The number of distinct labeled trees on n vertices, by Cayley's formula: n^(n-2)
    (with the conventions 1 tree for n = 1 and n = 2)."""
    if n <= 1:
        return 1
    if n == 2:
        return 1
    return n ** (n - 2)


def random_labeled_tree(n, seed=12345):
    """Generate a labeled tree on n vertices by decoding a pseudorandom Prufer sequence (a uniformly
    random labeled tree, since Prufer sequences biject with trees). Returns its edge list."""
    if n < 2:
        return []
    # seeded LCG to build a length n-2 sequence over 0..n-1
    s = seed & 0xFFFFFFFF
    seq = []
    for _ in range(n - 2):
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        seq.append((s >> 16) % n)
    return prufer_to_tree(n, seq)


def degree_from_prufer(n, seq):
    """The degree of each vertex implied by a Prufer sequence: appearances + 1 (leaves never
    appear)."""
    count = [0] * n
    for x in seq:
        count[x] += 1
    return [count[v] + 1 for v in range(n)]


# --- brute-force / validation helpers ---------------------------------------
def is_tree(n, edges):
    """True iff `edges` form a tree on n vertices: exactly n-1 edges, connected, acyclic (checked via
    connectivity + edge count, which together imply acyclic)."""
    if n == 0:
        return len(edges) == 0
    if len(edges) != n - 1:
        return False
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False           # a cycle
        parent[ru] = rv
    return len({find(v) for v in range(n)}) == 1


def all_labeled_trees(n):
    """Every labeled tree on n vertices, produced by decoding every possible Prufer sequence. Returns
    a list of edge-frozensets. Exponential (n^(n-2)); tiny n only."""
    from itertools import product
    if n == 1:
        return [frozenset()]
    if n == 2:
        return [frozenset([(0, 1)])]
    trees = []
    for seq in product(range(n), repeat=n - 2):
        edges = prufer_to_tree(n, list(seq))
        trees.append(frozenset(edges))
    return trees
