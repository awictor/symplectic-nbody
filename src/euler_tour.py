"""Euler tour of a tree: flatten a tree into an array so subtree and ancestor queries become ranges.

A rooted tree has no natural linear order, which makes "sum over this subtree" or "is u an ancestor of
v?" awkward. The EULER TOUR technique fixes that with a single DFS that records, for each node, the time
it is first ENTERED (tin) and finally LEFT (tout). The magic is that a node's entire subtree occupies a
CONTIGUOUS interval [tin[v], tout[v]] of tour time -- because DFS fully explores a subtree before
backing out. So a whole family of tree questions collapses to array questions:

  * u is an ANCESTOR of v  <=>  tin[u] <= tin[v] and tout[v] <= tout[u]   (u's interval contains v's)
  * the SUBTREE of v is exactly the nodes whose entry time lies in [tin[v], tout[v]]
  * a SUBTREE SUM/UPDATE becomes a RANGE sum/update on the flattened array -- pair the tour with a
    Fenwick tree or segment tree and subtree aggregates are O(log n)

This is the workhorse behind subtree updates in competitive programming and the tin/tout ordering used
by many tree algorithms. This module computes the Euler tour (entry/exit times and the flattened node
order) by an iterative DFS, answers ancestor and subtree-membership queries, maps a node to its subtree
range, and computes subtree sums over per-node values via a prefix sum on the tour. It is validated: the
subtree of every node is a contiguous tour interval of exactly its size; the ancestor test agrees with a
brute-force path-to-root check for all pairs; the root is an ancestor of everyone and the size of its
subtree is n; subtree sums match a brute-force sum over the actual descendants; entry times are a
permutation of 0..n-1; and it works on paths, stars, and random trees. Pure stdlib; the tree-flattening
companion to the LCA, Fenwick-tree, and sparse-table tools."""

from __future__ import annotations


def euler_tour(n, edges, root=0):
    """Compute the Euler tour of a rooted tree. Returns a dict with tin, tout, order, parent, depth.

    tin[v]/tout[v]: entry/exit times (the subtree of v is the entry-time interval [tin[v], tout[v]]);
    order: nodes in entry-time order; parent/depth: standard tree info."""
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    tin = [0] * n
    tout = [0] * n
    parent = [-1] * n
    depth = [0] * n
    order = []
    timer = 0
    # iterative DFS with an explicit child-iterator stack
    stack = [(root, -1, iter(adj[root]))]
    tin[root] = timer
    order.append(root)
    timer += 1
    seen = [False] * n
    seen[root] = True
    while stack:
        u, par, it = stack[-1]
        advanced = False
        for w in it:
            if not seen[w]:
                seen[w] = True
                parent[w] = u
                depth[w] = depth[u] + 1
                tin[w] = timer
                order.append(w)
                timer += 1
                stack.append((w, u, iter(adj[w])))
                advanced = True
                break
        if not advanced:
            tout[u] = timer - 1               # last entry time within u's subtree
            stack.pop()
    return {"tin": tin, "tout": tout, "order": order, "parent": parent, "depth": depth}


def is_ancestor(tour, u, v):
    """True if u is an ancestor of v (u == v counts as ancestor)."""
    return tour["tin"][u] <= tour["tin"][v] and tour["tout"][v] <= tour["tout"][u]


def subtree_range(tour, v):
    """The [start, end] entry-time interval occupied by v's subtree (inclusive)."""
    return tour["tin"][v], tour["tout"][v]


def subtree_size(tour, v):
    """Number of nodes in v's subtree."""
    return tour["tout"][v] - tour["tin"][v] + 1


def subtree_nodes(tour, v):
    """The nodes in v's subtree, read straight off the tour order."""
    lo, hi = subtree_range(tour, v)
    return tour["order"][lo:hi + 1]


def subtree_sum(tour, values, v):
    """Sum of `values[node]` over every node in v's subtree, using a prefix sum on the tour."""
    lo, hi = subtree_range(tour, v)
    # values indexed by node; lay them out in tour order
    ordered = [values[node] for node in tour["order"]]
    prefix = [0.0] * (len(ordered) + 1)
    for i, x in enumerate(ordered):
        prefix[i + 1] = prefix[i] + x
    return prefix[hi + 1] - prefix[lo]


def brute_subtree_nodes(n, edges, v, root=0):
    """Reference: the descendants of v (including v) by explicit BFS from the rooted tree."""
    adj = [[] for _ in range(n)]
    for (a, b) in edges:
        adj[a].append(b)
        adj[b].append(a)
    parent = [-1] * n
    order = []
    stack = [root]
    seen = [False] * n
    seen[root] = True
    while stack:
        u = stack.pop()
        order.append(u)
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                parent[w] = u
                stack.append(w)
    # collect the subtree of v
    sub = set([v])
    changed = True
    # repeatedly add children of nodes already in sub
    child = [[] for _ in range(n)]
    for x in range(n):
        if parent[x] != -1:
            child[parent[x]].append(x)
    stack = [v]
    result = set()
    while stack:
        u = stack.pop()
        result.add(u)
        for c in child[u]:
            stack.append(c)
    return result
