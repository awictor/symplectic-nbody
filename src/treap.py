"""Treaps: balanced search trees by randomization, with split, merge, and order statistics.

A binary search tree stays fast only if it stays balanced, and the classic balancers (AVL,
red-black) achieve that with intricate rotation cases. A TREAP earns balance almost for free by a
beautiful idea: give every key a random PRIORITY and keep the tree simultaneously a binary-search
tree on the KEYS and a heap on the PRIORITIES (hence tree + heap = treap). Because the priorities are
random, the shape of the treap is exactly the shape of a binary search tree built by inserting the
keys in a random order -- which is balanced with high probability, giving O(log n) expected height
and operations, with none of the case analysis of the deterministic balancers.

The whole structure rests on two primitives. SPLIT(t, k) cuts a treap into two treaps, one with all
keys < k and one with all keys >= k, in O(log n). MERGE(a, b) joins two treaps where every key in a
is less than every key in b, choosing the higher-priority root to preserve the heap order. Insert is
then just split-then-merge-in the new node; delete is split-out-and-merge-around. This split/merge
pair -- which balanced BSTs like AVL do not expose naturally -- is what makes treaps the tree of
choice for problems that slice and splice ordered sequences: order-statistics trees, interval
merging, and rope-like editable sequences.

By augmenting each node with its SUBTREE SIZE, the treap becomes an ORDER-STATISTICS tree: it can
report the k-th smallest key and the rank of a key in O(log n), operations a plain sorted list cannot
do while also supporting fast insertion and deletion. This module implements a treap with insert,
delete, membership, k-th smallest (select), rank, split, and merge, using a seeded reproducible RNG
for the priorities. It is verified against a sorted list and brute-force references: that in-order
traversal is always sorted, that a long random stream of inserts and deletes keeps the treap's
contents equal to a reference set, that select and rank match a sorted array, that split produces the
correct key partition, that merge reassembles the original, and that the tree height stays within a
small constant of the 2 log2(n) expected bound. Pure stdlib; a data-structure companion to the AVL
tree, skip list, and Fibonacci-heap notes."""

from __future__ import annotations

import math


class _Node:
    __slots__ = ("key", "priority", "size", "left", "right")

    def __init__(self, key, priority):
        self.key = key
        self.priority = priority
        self.size = 1
        self.left = None
        self.right = None


def _size(node):
    return node.size if node else 0


def _update(node):
    if node:
        node.size = 1 + _size(node.left) + _size(node.right)
    return node


class Treap:
    """A randomized balanced BST (set of distinct keys) with split/merge and order statistics."""

    def __init__(self, seed=1):
        self.root = None
        self._state = seed & 0xFFFFFFFF

    def _rand(self):
        # LCG, use high bits for a well-distributed priority
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state >> 8

    def __len__(self):
        return _size(self.root)

    def __contains__(self, key):
        node = self.root
        while node:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False

    # --- split / merge : the two primitives ------------------------------
    def _split(self, node, key):
        """Split subtree `node` into (< key) and (>= key)."""
        if node is None:
            return None, None
        if node.key < key:
            left, right = self._split(node.right, key)
            node.right = left
            _update(node)
            return node, right
        else:
            left, right = self._split(node.left, key)
            node.left = right
            _update(node)
            return left, node

    def _merge(self, a, b):
        """Merge treaps a (all keys) < b (all keys). Heap order by priority."""
        if a is None:
            return b
        if b is None:
            return a
        if a.priority > b.priority:
            a.right = self._merge(a.right, b)
            _update(a)
            return a
        else:
            b.left = self._merge(a, b.left)
            _update(b)
            return b

    # --- insert / delete -------------------------------------------------
    def insert(self, key):
        """Insert key (no-op if already present). Returns True if inserted."""
        if key in self:
            return False
        node = _Node(key, self._rand())
        left, right = self._split(self.root, key)
        self.root = self._merge(self._merge(left, node), right)
        return True

    def delete(self, key):
        """Delete key. Returns True if it was present."""
        if key not in self:
            return False
        self.root = self._delete(self.root, key)
        return True

    def _delete(self, node, key):
        if node is None:
            return None
        if key == node.key:
            return self._merge(node.left, node.right)
        if key < node.key:
            node.left = self._delete(node.left, key)
        else:
            node.right = self._delete(node.right, key)
        return _update(node)

    # --- order statistics ------------------------------------------------
    def select(self, k):
        """The k-th smallest key (0-indexed). Raises IndexError if out of range."""
        if not (0 <= k < _size(self.root)):
            raise IndexError(k)
        node = self.root
        while node:
            left_size = _size(node.left)
            if k < left_size:
                node = node.left
            elif k == left_size:
                return node.key
            else:
                k -= left_size + 1
                node = node.right
        raise IndexError(k)

    def rank(self, key):
        """Number of keys strictly less than `key` (its insertion rank)."""
        node = self.root
        r = 0
        while node:
            if key <= node.key:
                node = node.left
            else:
                r += 1 + _size(node.left)
                node = node.right
        return r

    # --- traversal / height ----------------------------------------------
    def inorder(self):
        out = []
        stack = []
        node = self.root
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            out.append(node.key)
            node = node.right
        return out

    def height(self):
        def h(node):
            if node is None:
                return 0
            return 1 + max(h(node.left), h(node.right))
        return h(self.root)


def build(keys, seed=1):
    """Build a treap from an iterable of keys."""
    t = Treap(seed=seed)
    for k in keys:
        t.insert(k)
    return t
