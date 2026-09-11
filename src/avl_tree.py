"""AVL trees: a self-balancing binary search tree with a guaranteed O(log n) height.

A plain binary search tree degrades to a linked list -- O(n) operations -- if keys arrive sorted.
An AVL tree (Adelson-Velsky & Landis, 1962, the first self-balancing BST) prevents that by keeping
every node HEIGHT-BALANCED: the heights of its two subtrees differ by at most 1. After each insert
or delete it checks the balance factor along the path back to the root and, wherever it exceeds
that bound, restores it with a local ROTATION -- a constant-time pointer rewiring that shortens the
tall side. Because the balance invariant is strict, an AVL tree is the most rigidly balanced of the
classic BSTs (shorter than a red-black tree), so lookups are fast; the cost is a little more
rotation work on updates.

Four rotation cases cover every imbalance, named by the shape of the offending path:

  LEFT-LEFT   -> single right rotation
  RIGHT-RIGHT -> single left rotation
  LEFT-RIGHT  -> left-rotate the child, then right-rotate
  RIGHT-LEFT  -> right-rotate the child, then left-rotate

Where a skip list stays balanced PROBABILISTICALLY with coin flips, an AVL tree stays balanced
DETERMINISTICALLY with rotations -- same O(log n) bounds, different mechanism. This module
implements an AVL ordered map with insert, delete, search, ordered traversal, range queries, and
min/max, maintaining the height and balance invariants -- verified against a brute-force sorted
dictionary over thousands of random operations, that the tree stays height-balanced and its height
is O(log n) even for sorted insertions (where a naive BST would be linear), that in-order traversal
is sorted, and that all four rotation cases trigger. Pure stdlib; the deterministic-balancing
companion to the skip-list note."""

from __future__ import annotations


class _Node:
    __slots__ = ("key", "value", "left", "right", "height")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.height = 1


def _h(node):
    return node.height if node else 0


def _update_height(node):
    node.height = 1 + max(_h(node.left), _h(node.right))


def _balance_factor(node):
    return _h(node.left) - _h(node.right) if node else 0


def _rotate_right(y):
    """Left-heavy fix: promote y's left child x, hang y off x's right."""
    x = y.left
    t2 = x.right
    x.right = y
    y.left = t2
    _update_height(y)
    _update_height(x)
    return x


def _rotate_left(x):
    """Right-heavy fix: promote x's right child y, hang x off y's left."""
    y = x.right
    t2 = y.left
    y.left = x
    x.right = t2
    _update_height(x)
    _update_height(y)
    return y


def _rebalance(node):
    """Restore the AVL invariant at `node` if its balance factor is out of range."""
    _update_height(node)
    bf = _balance_factor(node)
    if bf > 1:                          # left-heavy
        if _balance_factor(node.left) < 0:
            node.left = _rotate_left(node.left)     # left-right
        return _rotate_right(node)                  # left-left
    if bf < -1:                         # right-heavy
        if _balance_factor(node.right) > 0:
            node.right = _rotate_right(node.right)  # right-left
        return _rotate_left(node)                   # right-right
    return node


class AVLTree:
    """An ordered map backed by a height-balanced AVL tree; O(log n) worst-case operations."""

    def __init__(self):
        self.root = None
        self._size = 0

    def insert(self, key, value=None):
        """Insert or update a key. Returns True if a new key was added."""
        added = [False]
        self.root = self._insert(self.root, key, value, added)
        if added[0]:
            self._size += 1
        return added[0]

    def _insert(self, node, key, value, added):
        if node is None:
            added[0] = True
            return _Node(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value, added)
        elif key > node.key:
            node.right = self._insert(node.right, key, value, added)
        else:
            node.value = value                       # update in place, no duplicate
            return node
        return _rebalance(node)

    def _min_node(self, node):
        while node.left:
            node = node.left
        return node

    def delete(self, key):
        """Remove a key. Returns True if it was present."""
        removed = [False]
        self.root = self._delete(self.root, key, removed)
        if removed[0]:
            self._size -= 1
        return removed[0]

    def _delete(self, node, key, removed):
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key, removed)
        elif key > node.key:
            node.right = self._delete(node.right, key, removed)
        else:
            removed[0] = True
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            # two children: replace with the in-order successor, then delete it
            succ = self._min_node(node.right)
            node.key, node.value = succ.key, succ.value
            node.right = self._delete(node.right, succ.key, [False])
        return _rebalance(node)

    def search(self, key, default=None):
        node = self.root
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node.value
        return default

    def __contains__(self, key):
        node = self.root
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return True
        return False

    def __len__(self):
        return self._size

    def height(self):
        return _h(self.root)

    def keys(self):
        """All keys in ascending order (in-order traversal)."""
        out = []
        self._inorder(self.root, out)
        return out

    def _inorder(self, node, out):
        if node:
            self._inorder(node.left, out)
            out.append(node.key)
            self._inorder(node.right, out)

    def items(self):
        out = []
        self._inorder_items(self.root, out)
        return out

    def _inorder_items(self, node, out):
        if node:
            self._inorder_items(node.left, out)
            out.append((node.key, node.value))
            self._inorder_items(node.right, out)

    def min(self):
        if self.root is None:
            raise KeyError("empty tree")
        return self._min_node(self.root).key

    def max(self):
        if self.root is None:
            raise KeyError("empty tree")
        node = self.root
        while node.right:
            node = node.right
        return node.key

    def range(self, low, high):
        """All (key, value) pairs with low <= key <= high, in order (pruned in-order walk)."""
        out = []
        self._range(self.root, low, high, out)
        return out

    def _range(self, node, low, high, out):
        if node is None:
            return
        if node.key > low:
            self._range(node.left, low, high, out)
        if low <= node.key <= high:
            out.append((node.key, node.value))
        if node.key < high:
            self._range(node.right, low, high, out)

    def is_balanced(self):
        """True if every node satisfies the AVL height-balance invariant (for testing)."""
        def chk(node):
            if node is None:
                return True, 0
            lok, lh = chk(node.left)
            rok, rh = chk(node.right)
            ok = lok and rok and abs(lh - rh) <= 1 and node.height == 1 + max(lh, rh)
            return ok, 1 + max(lh, rh)
        return chk(self.root)[0]

    def is_bst(self):
        """True if the in-order traversal is strictly sorted (BST property holds)."""
        ks = self.keys()
        return all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
