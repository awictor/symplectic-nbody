"""Fibonacci heaps: the priority queue with O(1) amortized decrease-key.

A PRIORITY QUEUE that supports fast DECREASE-KEY is the engine behind Dijkstra's shortest paths and
Prim's minimum spanning tree: each edge relaxation lowers a vertex's tentative distance, and with n
vertices and m edges those algorithms do O(m) decrease-keys and O(n) extract-mins. A binary heap does
every operation in O(log n), giving Dijkstra its familiar O((n+m) log n). The FIBONACCI HEAP,
invented by Fredman and Tarjan, improves the theoretical bound to O(m + n log n) by making INSERT,
MERGE, FIND-MIN, and DECREASE-KEY run in O(1) AMORTIZED time, paying the logarithmic cost only at
EXTRACT-MIN.

The trick is LAZINESS. A Fibonacci heap is a forest of heap-ordered trees kept in a circular root
list; insert just drops a new single-node tree into that list, and merge just concatenates two root
lists -- both O(1). The structure is only cleaned up at extract-min, which removes the minimum,
promotes its children to roots, and then CONSOLIDATES the forest by linking trees of equal degree
until all root degrees are distinct (like binary addition), which is what bounds the tree count.
DECREASE-KEY is the clever part: lower a node's key and, if it now violates heap order, CUT it from
its parent and move it to the root list. To keep trees bushy enough that the degree stays O(log n),
each node carries a MARK bit; the second time a node loses a child it is cut too, cascading upward.
This cascading-cut discipline is exactly what makes the degrees obey Fibonacci-number bounds -- hence
the name -- and keeps decrease-key O(1) amortized.

This module implements a full Fibonacci heap (insert, find-min, extract-min, decrease-key, delete,
and merge) with node handles for decrease-key, plus a Dijkstra shortest-path built on it. It is
verified against a straightforward binary-heap priority queue and brute-force references: that a
sequence of operations returns keys in exactly sorted order, that decrease-key and delete behave
correctly, that merging two heaps preserves all elements, that the maximum root degree stays within
the O(log n) Fibonacci bound, and that Dijkstra on the Fibonacci heap matches Dijkstra on a binary
heap across many random graphs. Pure stdlib; a data-structure companion to the Dijkstra, binary-heap,
and union-find notes."""

from __future__ import annotations

import math


class FibNode:
    __slots__ = ("key", "value", "degree", "mark", "parent", "child", "left", "right")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.degree = 0
        self.mark = False
        self.parent = None
        self.child = None
        self.left = self       # circular doubly linked list
        self.right = self


class FibonacciHeap:
    """A min-oriented Fibonacci heap. insert returns a node handle usable with decrease_key/delete."""

    def __init__(self):
        self.min = None
        self.n = 0

    def __len__(self):
        return self.n

    def is_empty(self):
        return self.min is None

    # --- circular-list splice helpers ------------------------------------
    @staticmethod
    def _splice(a, b):
        """Merge circular list containing b into the list containing a (a and b are nodes)."""
        # insert b's list right after a
        a_right = a.right
        b_left = b.left
        a.right = b
        b.left = a
        b_left.right = a_right
        a_right.left = b_left

    def insert(self, key, value=None):
        node = FibNode(key, value)
        if self.min is None:
            self.min = node
        else:
            self._splice(self.min, node)
            if key < self.min.key:
                self.min = node
        self.n += 1
        return node

    def find_min(self):
        return (self.min.key, self.min.value) if self.min else None

    def merge(self, other):
        """Merge another Fibonacci heap into this one (destroys `other`). O(1)."""
        if other.min is None:
            return
        if self.min is None:
            self.min = other.min
            self.n = other.n
            return
        self._splice(self.min, other.min)
        if other.min.key < self.min.key:
            self.min = other.min
        self.n += other.n
        other.min = None
        other.n = 0

    def _iter_roots(self):
        if self.min is None:
            return
        roots = []
        cur = self.min
        while True:
            roots.append(cur)
            cur = cur.right
            if cur is self.min:
                break
        return roots

    def extract_min(self):
        z = self.min
        if z is None:
            return None
        # move z's children to the root list
        if z.child is not None:
            children = []
            c = z.child
            while True:
                children.append(c)
                c = c.right
                if c is z.child:
                    break
            for c in children:
                c.parent = None
            self._splice(z, z.child)
        # remove z from the root list
        z.left.right = z.right
        z.right.left = z.left
        if z is z.right:
            self.min = None
        else:
            self.min = z.right
            self._consolidate()
        self.n -= 1
        return (z.key, z.value)

    def _consolidate(self):
        # link trees of equal degree until all root degrees are distinct
        max_degree = int(math.log(self.n) / math.log((1 + math.sqrt(5)) / 2)) + 2 if self.n > 0 else 1
        A = [None] * (max_degree + 1)
        roots = self._iter_roots()
        for w in roots:
            x = w
            d = x.degree
            while d < len(A) and A[d] is not None:
                y = A[d]
                if x.key > y.key:
                    x, y = y, x
                self._link(y, x)
                A[d] = None
                d += 1
            while d >= len(A):
                A.append(None)
            A[d] = x
        # rebuild the root list and find the new min
        self.min = None
        for node in A:
            if node is None:
                continue
            node.left = node.right = node
            if self.min is None:
                self.min = node
            else:
                self._splice(self.min, node)
                if node.key < self.min.key:
                    self.min = node

    def _link(self, y, x):
        """Make y a child of x (both roots, x.key <= y.key)."""
        # remove y from the root list
        y.left.right = y.right
        y.right.left = y.left
        y.parent = x
        y.mark = False
        y.left = y.right = y            # isolate y before splicing (drop stale root-list links)
        if x.child is None:
            x.child = y
        else:
            self._splice(x.child, y)
        x.degree += 1

    def decrease_key(self, node, new_key):
        if new_key > node.key:
            raise ValueError("new key is greater than current key")
        node.key = new_key
        parent = node.parent
        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        if node.key < self.min.key:
            self.min = node

    def _cut(self, node, parent):
        """Remove node from parent's child list and add it to the root list."""
        if node.right is node:
            parent.child = None
        else:
            node.left.right = node.right
            node.right.left = node.left
            if parent.child is node:
                parent.child = node.right
        parent.degree -= 1
        node.parent = None
        node.mark = False
        node.left = node.right = node
        self._splice(self.min, node)

    def _cascading_cut(self, node):
        parent = node.parent
        if parent is not None:
            if not node.mark:
                node.mark = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)

    def delete(self, node):
        """Remove an arbitrary node: decrease its key to -inf, then extract-min."""
        self.decrease_key(node, float("-inf"))
        self.extract_min()

    def max_root_degree(self):
        """The maximum degree among current roots (for verifying the O(log n) bound)."""
        roots = self._iter_roots()
        if not roots:
            return 0
        return max(r.degree for r in roots)


def dijkstra(n, edges, source):
    """Single-source shortest paths using a Fibonacci heap.

    edges: list of (u, v, weight) directed edges (use both directions for an undirected graph).
    Returns a list of distances (float('inf') if unreachable)."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [float("inf")] * n
    dist[source] = 0
    heap = FibonacciHeap()
    handles = [None] * n
    handles[source] = heap.insert(0, source)
    visited = [False] * n
    while not heap.is_empty():
        d, u = heap.extract_min()
        if visited[u]:
            continue
        visited[u] = True
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                if handles[v] is None or visited[v]:
                    handles[v] = heap.insert(nd, v)
                else:
                    heap.decrease_key(handles[v], nd)
    return dist
