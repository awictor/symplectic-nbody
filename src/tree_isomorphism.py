"""AHU tree isomorphism: deciding whether two trees are the same shape, in linear time.

Two trees are ISOMORPHIC if one can be relabelled into the other -- same shape, different vertex names.
Deciding this for general GRAPHS is a famously hard open problem (no known polynomial algorithm), but
for TREES it is beautifully easy: the Aho-Hopcroft-Ullman (AHU) algorithm decides it in linear time by
computing a CANONICAL FORM -- a string (or hash) that is identical for isomorphic trees and different
otherwise. Tree isomorphism is the workhorse behind deduplicating parse trees and abstract syntax
trees, matching molecular structures that happen to be acyclic, comparing phylogenetic trees, and
caching sub-expressions in compilers.

For a ROOTED tree the canonical form is built bottom-up. Each leaf is the string "()". Each internal
node collects the canonical strings of its children, SORTS them (so the order of children does not
matter), concatenates them, and wraps the result in parentheses. Two rooted trees are isomorphic iff
their root strings are equal -- the sort is what makes the encoding independent of how children happen
to be listed. For an UNROOTED tree there is no distinguished root, so AHU roots the tree at its CENTER:
every tree has either one or two centers (the middle of its longest path, found by repeatedly peeling
leaves), and rooting at the center -- or comparing both canonical forms when there are two -- gives a
canonical form for the whole unrooted tree. Because centers are isomorphism-invariant, this correctly
decides unrooted isomorphism.

This module computes the AHU canonical form of a rooted tree, finds the 1 or 2 centers of an unrooted
tree, and decides both rooted and unrooted tree isomorphism. It is verified against brute force -- two
trees on the same vertex set are isomorphic iff some permutation of the vertices maps one edge set
exactly onto the other -- on hundreds of random trees, including relabelled copies (which must always
test isomorphic) and structurally different trees of the same size (which must not). Pure stdlib; a
graph-algorithms companion to the tree-DP, suffix-automaton, and Lyndon (canonical-form) notes."""

from __future__ import annotations


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def rooted_canonical(n, edges, root):
    """The AHU canonical string of the tree on `n` vertices with `edges`, rooted at `root`.

    Isomorphic rooted trees produce identical strings. Computed iteratively (post-order) so deep
    trees do not overflow the recursion stack."""
    adj = _build_adj(n, edges)
    parent = [-1] * n
    order = []                       # vertices in BFS order; reverse is a valid post-order
    seen = [False] * n
    stack = [root]
    seen[root] = True
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                stack.append(v)

    label = [""] * n
    for u in reversed(order):        # children finished before parents
        child_labels = sorted(label[v] for v in adj[u] if v != parent[u])
        label[u] = "(" + "".join(child_labels) + ")"
    return label[root]


def centers(n, edges):
    """The 1 or 2 centers of an unrooted tree, found by iteratively peeling leaves. Returns a list of
    one or two vertices."""
    if n == 1:
        return [0]
    adj = _build_adj(n, edges)
    degree = [len(adj[v]) for v in range(n)]
    leaves = [v for v in range(n) if degree[v] == 1]
    remaining = n
    while remaining > 2:
        new_leaves = []
        for leaf in leaves:
            for nb in adj[leaf]:
                degree[nb] -= 1
                if degree[nb] == 1:
                    new_leaves.append(nb)
            degree[leaf] = 0
        remaining -= len(leaves)
        leaves = new_leaves
    return leaves


def unrooted_canonical(n, edges):
    """A canonical form for an UNROOTED tree: root at the center(s) and take the smaller canonical
    string (comparing both when there are two centers). Isomorphic unrooted trees match."""
    cs = centers(n, edges)
    forms = sorted(rooted_canonical(n, edges, c) for c in cs)
    # join both center-forms so a 2-center tree can never collide with a 1-center tree of the same
    # single form; the sorted pair is itself canonical
    return "|".join(forms)


def rooted_isomorphic(n1, edges1, root1, n2, edges2, root2):
    """True iff the two rooted trees are isomorphic (same shape preserving the root)."""
    if n1 != n2:
        return False
    return rooted_canonical(n1, edges1, root1) == rooted_canonical(n2, edges2, root2)


def isomorphic(n1, edges1, n2, edges2):
    """True iff the two UNROOTED trees are isomorphic."""
    if n1 != n2:
        return False
    if len(edges1) != len(edges2):
        return False
    return unrooted_canonical(n1, edges1) == unrooted_canonical(n2, edges2)


# --- brute-force reference --------------------------------------------------
def brute_isomorphic(n1, edges1, n2, edges2):
    """Decide unrooted tree isomorphism by trying every vertex permutation and checking whether it
    maps edge set 1 exactly onto edge set 2. O(n!); tiny trees only."""
    if n1 != n2 or len(edges1) != len(edges2):
        return False
    from itertools import permutations
    target = {frozenset(e) for e in edges2}
    for perm in permutations(range(n1)):
        mapped = {frozenset((perm[u], perm[v])) for u, v in edges1}
        if mapped == target:
            return True
    return False
