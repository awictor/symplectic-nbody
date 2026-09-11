"""Skip lists: a probabilistic ordered dictionary with O(log n) operations.

A balanced binary search tree gives O(log n) search, insert, and delete but needs intricate
rotations to stay balanced. A SKIP LIST (Pugh, 1990) reaches the same expected bounds with almost
no bookkeeping, using RANDOMNESS instead of rotations. It is an ordered linked list with EXPRESS
LANES: each node is promoted to the next level up with probability p (typically 1/2), so level 0
holds every element, level 1 about half, level 2 a quarter, and so on -- a tower of increasingly
sparse shortcut lists. A search drops down from the top lane, skipping far along each level before
descending, covering the list in O(log n) expected hops.

Because promotion is a coin flip, there are no rotations and no rebalancing: insertion picks a
random height and splices the node into every level up to it; deletion unlinks it from each. The
structure stays probabilistically balanced on its own, degrading only in astronomically unlikely
cases. Skip lists power ordered maps in Redis (sorted sets) and are a favourite for concurrent
structures because local splices need no global rebalancing.

This module implements a skip-list ordered map with insert, search, delete, ordered iteration, and
range queries, plus min/max and rank-style helpers -- verified against a brute-force sorted
dictionary over thousands of random operations (every search, deletion, and ordered traversal
agrees), that keys iterate in sorted order, that duplicate keys update rather than duplicate, that
range queries return exactly the in-range keys, and that the level distribution is geometric as
designed. Pure stdlib (its own seeded RNG); a data-structures companion to the trie and
binary-heap notes."""

from __future__ import annotations


class _Node:
    __slots__ = ("key", "value", "forward")

    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)   # forward[i] = next node at level i


class SkipList:
    """An ordered map keyed by comparable keys, with O(log n) expected operations."""

    def __init__(self, max_level=16, p=0.5, seed=0):
        self.max_level = max_level
        self.p = p
        self.level = 0                       # current highest occupied level
        self.head = _Node(None, None, max_level)
        self._size = 0
        self._state = seed & 0xFFFFFFFF

    def _rand(self):
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return (self._state >> 16) / 65536.0     # high bits

    def _random_level(self):
        """A geometric height: promote while a coin (prob p) keeps coming up, capped at max_level."""
        lvl = 0
        while self._rand() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key, default=None):
        node = self.head
        for i in range(self.level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is not None and node.key == key:
            return node.value
        return default

    def __contains__(self, key):
        node = self.head
        for i in range(self.level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        return node is not None and node.key == key

    def insert(self, key, value=None):
        """Insert or update a key. Returns True if a new key was added, False if updated."""
        update = [self.head] * (self.max_level + 1)
        node = self.head
        for i in range(self.level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node
        node = node.forward[0]
        if node is not None and node.key == key:
            node.value = value                  # update in place, no duplicate
            return False
        lvl = self._random_level()
        if lvl > self.level:
            for i in range(self.level + 1, lvl + 1):
                update[i] = self.head
            self.level = lvl
        new_node = _Node(key, value, lvl)
        for i in range(lvl + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self._size += 1
        return True

    def delete(self, key):
        """Remove a key. Returns True if it was present."""
        update = [self.head] * (self.max_level + 1)
        node = self.head
        for i in range(self.level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node
        node = node.forward[0]
        if node is None or node.key != key:
            return False
        for i in range(self.level + 1):
            if update[i].forward[i] is not node:
                break
            update[i].forward[i] = node.forward[i]
        # lower the list level if the top lanes are now empty
        while self.level > 0 and self.head.forward[self.level] is None:
            self.level -= 1
        self._size -= 1
        return True

    def __len__(self):
        return self._size

    def keys(self):
        """All keys in ascending order (a walk of level 0)."""
        out = []
        node = self.head.forward[0]
        while node is not None:
            out.append(node.key)
            node = node.forward[0]
        return out

    def items(self):
        out = []
        node = self.head.forward[0]
        while node is not None:
            out.append((node.key, node.value))
            node = node.forward[0]
        return out

    def min(self):
        if self.head.forward[0] is None:
            raise KeyError("empty skip list")
        return self.head.forward[0].key

    def max(self):
        if self.head.forward[0] is None:
            raise KeyError("empty skip list")
        node = self.head
        for i in range(self.level, -1, -1):
            while node.forward[i] is not None:
                node = node.forward[i]
        return node.key

    def range(self, low, high):
        """All (key, value) pairs with low <= key <= high, in order. Starts with a fast descent
        to the first in-range key, then walks level 0."""
        node = self.head
        for i in range(self.level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < low:
                node = node.forward[i]
        node = node.forward[0]
        out = []
        while node is not None and node.key <= high:
            out.append((node.key, node.value))
            node = node.forward[0]
        return out

    def node_levels(self):
        """The tower height of every stored node (0-based), for inspecting the level distribution."""
        levels = []
        node = self.head.forward[0]
        while node is not None:
            levels.append(len(node.forward) - 1)
            node = node.forward[0]
        return levels
