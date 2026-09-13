"""Cartesian trees: the O(n) bridge between an array's range-minima and a tree's ancestors.

A CARTESIAN TREE built on a sequence is a binary tree with two simultaneous properties: its in-order
traversal returns the sequence unchanged (a BINARY SEARCH TREE on POSITION), and every node is smaller
than its children (a MIN-HEAP on VALUE). There is exactly one such tree for a sequence with distinct
values, and -- remarkably -- it can be built in LINEAR TIME with a single stack pass, no sorting.

Why it matters: the Cartesian tree makes the RANGE-MINIMUM-QUERY and LOWEST-COMMON-ANCESTOR problems
the SAME problem. The minimum of a[i..j] sits, by the heap property, at the LCA of nodes i and j in
the Cartesian tree; conversely the LCA of two tree nodes is the array minimum between them. This
equivalence is the heart of the classic O(n)-preprocessing / O(1)-query RMQ algorithm (reduce RMQ to
LCA on the Cartesian tree, then LCA to +/-1 RMQ on the Euler tour), and it is why Cartesian trees show
up in suffix-array LCP structures, treaps (a Cartesian tree on (key, random-priority) pairs), and
pattern matching.

The linear build uses a stack holding the RIGHTMOST SPINE of the tree so far. For each new element,
pop everything on the stack larger than it (those become its left subtree, in order), attach it as the
right child of whatever remains, and push it. Each element is pushed and popped at most once, so the
whole construction is O(n).

This module builds the Cartesian tree, exposes its structure, and answers range-minimum queries by
LCA on the tree. It is validated against independent references: the in-order traversal reproduces the
sequence and the heap property holds at every node (so the tree is the unique Cartesian tree); the
build matches a naive recursive O(n^2) construction; and range-minimum queries answered via
tree-LCA agree with the repo's sparse-table RMQ and with brute force over every subrange. Pure
stdlib; the array-to-tree companion to the sparse-table RMQ/LCA, the treap, and the suffix-array LCP
tools."""

from __future__ import annotations


def build_cartesian_tree(a):
    """Build the min-Cartesian tree of sequence `a` in O(n). Returns (root, left, right, parent).

    left[i]/right[i] are child indices (or -1); parent[i] is the parent (or -1 for the root).
    Nodes are the array indices 0..n-1.
    """
    n = len(a)
    left = [-1] * n
    right = [-1] * n
    parent = [-1] * n
    stack = []  # indices on the current rightmost spine, increasing value bottom->top

    for i in range(n):
        last = -1
        # pop everything larger than a[i]; they become i's left subtree chain
        while stack and a[stack[-1]] > a[i]:
            last = stack.pop()
        # i's left child is the last popped (root of the popped chain)
        if last != -1:
            left[i] = last
            parent[last] = i
        # i becomes the right child of the new stack top
        if stack:
            right[stack[-1]] = i
            parent[i] = stack[-1]
        stack.append(i)

    root = stack[0] if stack else -1
    return root, left, right, parent


def inorder(root, left, right):
    """In-order traversal of the tree; for a Cartesian tree this yields 0,1,...,n-1 (positions)."""
    order = []
    stack = []
    node = root
    while stack or node != -1:
        while node != -1:
            stack.append(node)
            node = left[node]
        node = stack.pop()
        order.append(node)
        node = right[node]
    return order


def is_heap_ordered(a, left, right, parent):
    """True if every node's value <= its children's values (the min-heap property)."""
    n = len(a)
    for i in range(n):
        if left[i] != -1 and a[left[i]] < a[i]:
            return False
        if right[i] != -1 and a[right[i]] < a[i]:
            return False
    return True


class CartesianRMQ:
    """Range-minimum queries on a sequence via LCA on its Cartesian tree.

    Uses an O(n) Euler-tour + sparse-table LCA under the hood (the repo's LCA), so preprocessing is
    O(n log n) and each query O(1). The point is the reduction: RMQ = LCA on the Cartesian tree.
    """

    def __init__(self, a):
        self.a = list(a)
        n = len(a)
        self.root, self.left, self.right, self.parent = build_cartesian_tree(a)
        # build an edge list of the tree for the LCA structure
        edges = []
        for i in range(n):
            if self.parent[i] != -1:
                edges.append((self.parent[i], i))
        from sparse_table import LCA
        self._lca = LCA(n, edges, root=self.root) if n > 0 else None

    def min_index(self, l, r):
        """Index of the minimum in a[l..r] (inclusive), via the LCA of l and r."""
        if l > r:
            l, r = r, l
        return self._lca.query(l, r)

    def min_value(self, l, r):
        return self.a[self.min_index(l, r)]


def build_naive(a):
    """Reference O(n^2) recursive Cartesian-tree build (root = min, recurse on halves)."""
    n = len(a)
    left = [-1] * n
    right = [-1] * n
    parent = [-1] * n

    def rec(lo, hi, par):
        if lo > hi:
            return -1
        # index of the minimum in a[lo..hi]
        m = lo
        for k in range(lo + 1, hi + 1):
            if a[k] < a[m]:
                m = k
        parent[m] = par
        left[m] = rec(lo, m - 1, m)
        right[m] = rec(m + 1, hi, m)
        return m

    root = rec(0, n - 1, -1)
    return root, left, right, parent


def brute_min_index(a, l, r):
    """Reference: index of the minimum in a[l..r]."""
    m = l
    for k in range(l + 1, r + 1):
        if a[k] < a[m]:
            m = k
    return m
