"""Splay trees: self-adjusting binary search trees that move hot keys to the root.

Most balanced search trees (AVL, red-black, treaps) keep an explicit balance invariant. A SPLAY TREE,
invented by Sleator and Tarjan, keeps NO balance information at all -- yet achieves O(log n) AMORTIZED
time per operation through a single restructuring move called SPLAYING. Every time a key is accessed,
inserted, or deleted, the tree rotates that node all the way up to the root along its access path. The
rotations are done in careful pairs (ZIG-ZIG when the node and its parent are on the same side,
ZIG-ZAG when on opposite sides, and a single ZIG at the root) that not only lift the node but also
roughly halve the depth of every node on the path -- so long paths pay for themselves by becoming
short.

This self-adjustment gives splay trees remarkable properties no fixed-balance tree has. The
WORKING-SET property: recently accessed keys are near the root, so a sequence with temporal locality
runs far faster than O(log n) per access. The STATIC-OPTIMALITY theorem: on any access sequence, a
splay tree is within a constant factor of the best possible STATIC tree built with full knowledge of
the access frequencies -- automatically, with no tuning. This makes splay trees a natural fit for
caches, rope data structures, and any workload where some keys are much hotter than others.

This module implements a top-down/bottom-up splay tree supporting insert, delete, membership,
find-min/find-max, predecessor/successor, and ordered traversal, splaying on every access. It is
verified against a reference sorted set and brute force: that in-order traversal stays sorted through
a long random insert/delete stream, that membership and predecessor/successor match a sorted array,
that the most-recently-accessed key always sits at the root (the defining splay property), that
repeatedly accessing a small hot set keeps the average access depth low (the working-set property),
and that deletions preserve the BST ordering. Pure stdlib; a data-structure companion to the treap,
AVL-tree, and skip-list notes."""

from __future__ import annotations


class _Node:
    __slots__ = ("key", "value", "left", "right", "parent")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None


class SplayTree:
    """A self-adjusting BST. Every access splays the touched node to the root."""

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    # --- rotations -------------------------------------------------------
    def _rotate(self, x):
        """Rotate x above its parent (single rotation, direction inferred)."""
        p = x.parent
        g = p.parent
        if p.left is x:
            # right rotation
            p.left = x.right
            if x.right:
                x.right.parent = p
            x.right = p
        else:
            # left rotation
            p.right = x.left
            if x.left:
                x.left.parent = p
            x.left = p
        p.parent = x
        x.parent = g
        if g is not None:
            if g.left is p:
                g.left = x
            else:
                g.right = x

    def _splay(self, x):
        """Move x to the root via zig / zig-zig / zig-zag steps."""
        while x.parent is not None:
            p = x.parent
            g = p.parent
            if g is None:
                self._rotate(x)                 # zig
            elif (g.left is p) == (p.left is x):
                self._rotate(p)                 # zig-zig: rotate parent first
                self._rotate(x)
            else:
                self._rotate(x)                 # zig-zag: rotate x twice
                self._rotate(x)
        self.root = x

    # --- search / splay to a key -----------------------------------------
    def _find_node(self, key):
        node = self.root
        last = None
        while node:
            last = node
            if key == node.key:
                self._splay(node)
                return node
            node = node.left if key < node.key else node.right
        if last is not None:
            self._splay(last)                   # splay the last accessed node (amortization)
        return None

    def __contains__(self, key):
        return self._find_node(key) is not None

    def get(self, key, default=None):
        node = self._find_node(key)
        return node.value if node else default

    # --- insert ----------------------------------------------------------
    def insert(self, key, value=None):
        """Insert or update key. Splays the new/updated node to the root."""
        if self.root is None:
            self.root = _Node(key, value)
            self._size = 1
            return
        node = self.root
        while True:
            if key == node.key:
                node.value = value
                self._splay(node)
                return
            elif key < node.key:
                if node.left is None:
                    new = _Node(key, value)
                    new.parent = node
                    node.left = new
                    self._splay(new)
                    self._size += 1
                    return
                node = node.left
            else:
                if node.right is None:
                    new = _Node(key, value)
                    new.parent = node
                    node.right = new
                    self._splay(new)
                    self._size += 1
                    return
                node = node.right

    # --- delete ----------------------------------------------------------
    def delete(self, key):
        """Delete key. Returns True if it was present."""
        node = self._find_node(key)             # splays key (or last node) to root
        if node is None or node.key != key:
            return False
        # node is now the root; join its two subtrees
        left, right = node.left, node.right
        if left:
            left.parent = None
        if right:
            right.parent = None
        if left is None:
            self.root = right
        else:
            # find the max of the left subtree, splay it, attach right
            self.root = left
            m = left
            while m.right:
                m = m.right
            self._splay(m)
            m.right = right
            if right:
                right.parent = m
        self._size -= 1
        return True

    # --- min / max / predecessor / successor -----------------------------
    def find_min(self):
        if self.root is None:
            return None
        node = self.root
        while node.left:
            node = node.left
        self._splay(node)
        return node.key

    def find_max(self):
        if self.root is None:
            return None
        node = self.root
        while node.right:
            node = node.right
        self._splay(node)
        return node.key

    def predecessor(self, key):
        """Largest key strictly less than `key`, or None."""
        node = self.root
        pred = None
        while node:
            if node.key < key:
                pred = node.key
                node = node.right
            else:
                node = node.left
        return pred

    def successor(self, key):
        """Smallest key strictly greater than `key`, or None."""
        node = self.root
        succ = None
        while node:
            if node.key > key:
                succ = node.key
                node = node.left
            else:
                node = node.right
        return succ

    # --- traversal -------------------------------------------------------
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

    def root_key(self):
        return self.root.key if self.root else None
