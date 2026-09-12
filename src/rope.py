"""The rope -- a balanced tree that makes huge strings cheap to concatenate, split, and edit.

A text editor holding a hundred-megabyte file cannot store it as one flat Python string: inserting a
character in the middle would copy the entire buffer, O(n) per keystroke. The ROPE (Boehm, Atkinson &
Plass, 1995) is the data structure that fixes this. It represents a string as a balanced binary tree
whose LEAVES hold short substrings and whose INTERNAL nodes just record the total length of their left
subtree. The actual text lives only in the leaves; the tree is a scaffold that lets you find, split, and
join without ever copying the bulk of the characters.

The payoff is in the operations. CONCATENATION of two ropes is O(1) in the ideal case -- make a new root
whose children are the two ropes -- versus O(n) for flat strings. INDEXING walks down from the root using
the stored left-lengths to decide which way to go, O(log n) on a balanced tree. SPLIT at a position cuts
the tree into everything before and everything after in O(log n), and INSERT and DELETE are then just
split-and-concat. Because edits touch only O(log n) nodes on a path, an editor can insert into the middle
of a giant document in logarithmic time, and structural sharing means an undo history can keep old
versions cheaply.

Balance matters: naive concatenation can build a degenerate linked list that degrades every operation to
O(n). This implementation keeps the tree healthy by REBALANCING when a subtree gets too tall relative to
its size (a weight-balance condition), rebuilding that subtree from its leaf sequence, so depth stays
logarithmic. It provides construction from a string, length, character indexing, substring extraction,
concatenation, split, insert, delete, and materialisation back to a flat string, plus the tree height so
the balancing can be observed.

Pure standard library. The whole point is that the rope must behave EXACTLY like the string it
represents, only faster for edits, so that is what the tests check.

Validation. A rope is exercised alongside a plain Python string as an oracle: over thousands of seeded
random operations -- insert at a random position, delete a random range, concatenate, split and rejoin --
the rope's flattened contents must equal the reference string at every step, and random index and
substring queries must return the same characters. Concatenation length is additive; split then concat
is the identity; indexing out of range raises; and the tree stays BALANCED (height O(log n), checked to
be within a small constant factor of log2(length)) even after long sequences of edits that would make a
naive rope degenerate. Empty ropes and single characters are handled."""

import math


_LEAF_MAX = 8          # leaves hold at most this many characters
_BALANCE_RATIO = 3     # rebalance a subtree if its height exceeds this * ideal


class _Node:
    __slots__ = ("text", "left", "right", "weight", "length", "height")

    def __init__(self, text=None, left=None, right=None):
        if text is not None:
            # leaf
            self.text = text
            self.left = None
            self.right = None
            self.weight = len(text)
            self.length = len(text)
            self.height = 1
        else:
            # internal
            self.text = None
            self.left = left
            self.right = right
            self.weight = left.length if left else 0
            self.length = (left.length if left else 0) + (right.length if right else 0)
            self.height = 1 + max(left.height if left else 0, right.height if right else 0)

    def is_leaf(self):
        return self.text is not None


class Rope:
    """An immutable-ish rope; edit operations return new ropes sharing structure where possible."""

    def __init__(self, source=""):
        if isinstance(source, _Node):
            self.root = source
        else:
            self.root = _build_leaves(source)

    def __len__(self):
        return self.root.length if self.root else 0

    @property
    def height(self):
        return self.root.height if self.root else 0

    # -- indexing -----------------------------------------------------------
    def index(self, i):
        """The character at position i (O(log n))."""
        if i < 0 or i >= len(self):
            raise IndexError("rope index out of range")
        node = self.root
        while not node.is_leaf():
            if i < node.weight:
                node = node.left
            else:
                i -= node.weight
                node = node.right
        return node.text[i]

    def __getitem__(self, i):
        return self.index(i)

    # -- concatenation ------------------------------------------------------
    def concat(self, other):
        """Concatenate two ropes (O(1) amortised, then rebalanced if lopsided)."""
        if len(self) == 0:
            return Rope(other.root)
        if len(other) == 0:
            return Rope(self.root)
        root = _Node(left=self.root, right=other.root)
        return Rope(_maybe_rebalance(root))

    def __add__(self, other):
        return self.concat(other)

    # -- split --------------------------------------------------------------
    def split(self, i):
        """Split into (left rope of length i, right rope of the rest)."""
        if i < 0 or i > len(self):
            raise IndexError("split position out of range")
        left, right = _split_node(self.root, i)
        return Rope(left) if left else Rope(""), Rope(right) if right else Rope("")

    # -- insert / delete ----------------------------------------------------
    def insert(self, i, s):
        """Insert string (or rope) s at position i, returning a new rope."""
        left, right = self.split(i)
        mid = s if isinstance(s, Rope) else Rope(s)
        return left.concat(mid).concat(right)

    def delete(self, start, end):
        """Delete the half-open range [start, end), returning a new rope."""
        if not (0 <= start <= end <= len(self)):
            raise IndexError("delete range out of bounds")
        left, rest = self.split(start)
        _, right = rest.split(end - start)
        return left.concat(right)

    def substring(self, start, end):
        """The substring [start, end) as a new rope."""
        if not (0 <= start <= end <= len(self)):
            raise IndexError("substring range out of bounds")
        _, rest = self.split(start)
        mid, _ = rest.split(end - start)
        return mid

    # -- materialise --------------------------------------------------------
    def to_string(self):
        parts = []
        _collect_leaves(self.root, parts)
        return "".join(parts)

    def __str__(self):
        return self.to_string()

    def is_balanced(self):
        """True if the tree height is within a small factor of the ideal log2(length)."""
        n = len(self)
        if n <= 1:
            return True
        return self.height <= _BALANCE_RATIO * (math.log2(n) + 1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_leaves(s):
    """Build a balanced rope from a flat string by recursively halving into leaf chunks."""
    if len(s) <= _LEAF_MAX:
        return _Node(text=s)
    mid = len(s) // 2
    return _Node(left=_build_leaves(s[:mid]), right=_build_leaves(s[mid:]))


def _collect_leaves(node, out):
    if node is None:
        return
    if node.is_leaf():
        out.append(node.text)
    else:
        _collect_leaves(node.left, out)
        _collect_leaves(node.right, out)


def _split_node(node, i):
    """Split a node's subtree at index i; return (left_node_or_None, right_node_or_None)."""
    if node is None:
        return None, None
    if node.is_leaf():
        if i <= 0:
            return None, node
        if i >= node.length:
            return node, None
        return _Node(text=node.text[:i]), _Node(text=node.text[i:])
    # internal node
    if i < node.weight:
        left_left, left_right = _split_node(node.left, i)
        right = _join(left_right, node.right)
        return left_left, right
    elif i > node.weight:
        right_left, right_right = _split_node(node.right, i - node.weight)
        left = _join(node.left, right_left)
        return left, right_right
    else:
        return node.left, node.right


def _join(a, b):
    """Concatenate two nodes (either may be None)."""
    if a is None:
        return b
    if b is None:
        return a
    return _maybe_rebalance(_Node(left=a, right=b))


def _maybe_rebalance(node):
    """Rebuild the subtree from its leaves if it has become too tall for its size."""
    n = node.length
    ideal = math.log2(n) + 1 if n > 1 else 1
    if node.height > _BALANCE_RATIO * ideal:
        parts = []
        _collect_leaves(node, parts)
        return _build_leaves("".join(parts))
    return node
