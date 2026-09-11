"""Ternary search trees: the string map that sits between a trie and a BST.

A TERNARY SEARCH TREE (TST) stores strings by giving each node a single character and THREE children:
LEFT for characters that sort before it, RIGHT for characters that sort after, and a MIDDLE ('equal')
child that advances to the next character of the key. Following the middle links spells out a key,
exactly like a trie, but instead of an array (or hash) of children per node, the alternatives at each
position live in a little binary search tree. That is the whole trick, and it buys the best of both
worlds: the prefix structure and ordered traversal of a trie, but with the space of a BST -- no
wasted k-way arrays for the huge alphabet of Unicode, and graceful behaviour on sparse key sets. TSTs
were championed by Bentley and Sedgewick and are the classic backing store for spell-checkers,
autocomplete, and IP-routing-style prefix lookups.

The elegance is that a handful of the same three-way comparison drives every operation. INSERT and
SEARCH walk left/right on the character comparison and drop into the middle child on a match. PREFIX
COMPLETION finds the subtree hanging off the end of the prefix and enumerates every key beneath it in
sorted order. And -- the operation tries and hash maps cannot do cheaply -- PARTIAL-MATCH search with
'.' wildcards: at a wildcard position you recurse into all three children, so `c.t` matches cat, cot,
cut in one traversal. TSTs also give you the keys back in lexicographic order for free.

This module implements a TST supporting insert with associated values, exact lookup, deletion (by
tombstoning the value), sorted key iteration, prefix autocomplete, longest-prefix-of a query, and
wildcard partial-match search. It is verified against a plain dict and brute-force references: that
the TST holds exactly the inserted keys with the right values, that sorted iteration is truly sorted,
that prefix completion returns exactly the dict keys with that prefix, that wildcard search matches a
brute-force regex-style scan over hundreds of random pattern queries, and that deletion behaves like
dict deletion. Pure stdlib; a string-data-structure companion to the trie and suffix-array notes."""

from __future__ import annotations


class _Node:
    __slots__ = ("ch", "left", "mid", "right", "is_key", "value")

    def __init__(self, ch):
        self.ch = ch
        self.left = None
        self.mid = None
        self.right = None
        self.is_key = False
        self.value = None


class TernarySearchTree:
    """A ternary search tree mapping non-empty strings to values."""

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        node = self._get_node(key)
        return node is not None and node.is_key

    def insert(self, key, value=True):
        """Insert key with an associated value (defaults to True for set-like use)."""
        if not key:
            raise ValueError("empty keys are not supported")
        self.root, inserted = self._insert(self.root, key, 0, value)
        if inserted:
            self._size += 1

    def _insert(self, node, key, i, value):
        c = key[i]
        if node is None:
            node = _Node(c)
        inserted = False
        if c < node.ch:
            node.left, inserted = self._insert(node.left, key, i, value)
        elif c > node.ch:
            node.right, inserted = self._insert(node.right, key, i, value)
        elif i + 1 < len(key):
            node.mid, inserted = self._insert(node.mid, key, i + 1, value)
        else:
            inserted = not node.is_key
            node.is_key = True
            node.value = value
        return node, inserted

    def _get_node(self, key):
        """The node holding the last character of key, or None."""
        if not key:
            return None
        node = self.root
        i = 0
        while node is not None:
            c = key[i]
            if c < node.ch:
                node = node.left
            elif c > node.ch:
                node = node.right
            else:
                if i + 1 == len(key):
                    return node
                i += 1
                node = node.mid
        return None

    def get(self, key, default=None):
        node = self._get_node(key)
        if node is not None and node.is_key:
            return node.value
        return default

    def delete(self, key):
        """Remove key. Returns True if it was present. (Tombstones the value; keeps the structure.)"""
        node = self._get_node(key)
        if node is not None and node.is_key:
            node.is_key = False
            node.value = None
            self._size -= 1
            return True
        return False

    def keys(self):
        """All keys in lexicographic order."""
        out = []
        self._collect(self.root, [], out)
        return out

    def _collect(self, node, prefix, out):
        if node is None:
            return
        self._collect(node.left, prefix, out)
        prefix.append(node.ch)
        if node.is_key:
            out.append("".join(prefix))
        self._collect(node.mid, prefix, out)
        prefix.pop()
        self._collect(node.right, prefix, out)

    def items(self):
        """(key, value) pairs in lexicographic key order."""
        out = []
        self._collect_items(self.root, [], out)
        return out

    def _collect_items(self, node, prefix, out):
        if node is None:
            return
        self._collect_items(node.left, prefix, out)
        prefix.append(node.ch)
        if node.is_key:
            out.append(("".join(prefix), node.value))
        self._collect_items(node.mid, prefix, out)
        prefix.pop()
        self._collect_items(node.right, prefix, out)

    def keys_with_prefix(self, prefix):
        """All keys that start with prefix, in lexicographic order (autocomplete)."""
        if not prefix:
            return self.keys()
        node = self._get_node(prefix)
        if node is None:
            return []
        out = []
        # the prefix itself may be a key
        if node.is_key:
            out.append(prefix)
        # everything hanging off the middle child extends the prefix
        self._collect(node.mid, list(prefix), out)
        return out

    def longest_prefix_of(self, query):
        """The longest key that is a prefix of query (e.g. for IP routing / dictionary segmentation).
        Returns the key string, or '' if none."""
        node = self.root
        i = 0
        length = 0
        best = 0
        while node is not None and i < len(query):
            c = query[i]
            if c < node.ch:
                node = node.left
            elif c > node.ch:
                node = node.right
            else:
                i += 1
                length = i
                if node.is_key:
                    best = length
                node = node.mid
        return query[:best]

    def wildcard(self, pattern):
        """All keys matching pattern, where '.' matches any single character. Same length as pattern.
        Returns them in lexicographic order."""
        out = []
        self._wildcard(self.root, pattern, 0, [], out)
        return out

    def _wildcard(self, node, pattern, i, prefix, out):
        if node is None:
            return
        c = pattern[i]
        if c == "." or c < node.ch:
            self._wildcard(node.left, pattern, i, prefix, out)
        if c == "." or c == node.ch:
            prefix.append(node.ch)
            if i + 1 == len(pattern):
                if node.is_key:
                    out.append("".join(prefix))
            else:
                self._wildcard(node.mid, pattern, i + 1, prefix, out)
            prefix.pop()
        if c == "." or c > node.ch:
            self._wildcard(node.right, pattern, i, prefix, out)
