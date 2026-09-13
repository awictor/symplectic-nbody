"""Red-black trees -- the self-balancing search tree behind std::map, TreeMap, and the Linux kernel.

A binary search tree gives O(log n) lookups only if it stays balanced; feed a plain BST sorted data and
it degenerates into a linked list, O(n) per operation. The RED-BLACK TREE (Bayer 1972, Guibas-Sedgewick
1978) keeps itself balanced through insertions and deletions using a single bit of colour per node and a
handful of local rotations, guaranteeing the height never exceeds 2 log2(n+1). It is the balanced tree
that industry actually ships: C++'s std::map and std::set, Java's TreeMap and TreeSet, and the Linux
kernel's process scheduler and virtual-memory areas are all red-black trees, chosen over AVL because its
looser balance means far fewer rotations on update-heavy workloads.

The balance comes from four invariants coloured onto the nodes: (1) every node is RED or BLACK; (2) the
root is black; (3) a red node's children are both black (no two reds in a row); (4) every root-to-null
path passes through the same number of black nodes (the "black height"). Together these force the
longest path to be at most twice the shortest, so the tree is balanced without the strict height
tracking AVL needs. Insertion adds a red leaf and repairs any red-red violation by recolouring and
rotating up the tree; deletion is the intricate converse, fixing a "double-black" deficit by borrowing
blackness from a sibling or pushing it up. Each fix is local and O(1), so both operations are O(log n)
with at most a constant number of rotations.

This implementation augments every node with its SUBTREE SIZE, turning the tree into an ORDER-STATISTIC
tree: it answers "what is the k-th smallest key?" (select) and "how many keys are less than x?" (rank)
in O(log n), the operations that make a balanced BST a ranked, ordered dictionary. It provides insert,
delete, search, ordered iteration, min/max, predecessor/successor, select, and rank. Pure standard
library.

Validation. The tree is exercised alongside Python's ``dict`` and ``sorted`` as oracles over thousands
of seeded random insert/delete/search operations: it agrees on every membership and value, and an
in-order traversal exactly equals the sorted keys at every step. The RED-BLACK INVARIANTS are checked
directly after every operation -- root black, no red node with a red child, and equal black height on
all root-to-null paths -- and the height is verified to stay within the 2 log2(n+1) bound. The
order-statistic queries are checked against the sorted key list: select(k) equals the k-th sorted key
and rank(x) equals its position, over many random queries. Deleting every key empties the tree while
preserving invariants throughout."""

RED = True
BLACK = False


class _Node:
    __slots__ = ("key", "value", "color", "left", "right", "parent", "size")

    def __init__(self, key, value, color, nil):
        self.key = key
        self.value = value
        self.color = color
        self.left = nil
        self.right = nil
        self.parent = nil
        self.size = 1


class RedBlackTree:
    """An ordered, order-statistic map backed by a red-black tree."""

    def __init__(self):
        # sentinel NIL node: black, size 0, its own family
        self.nil = _Node(None, None, BLACK, None)
        self.nil.size = 0
        self.nil.left = self.nil.right = self.nil.parent = self.nil
        self.root = self.nil

    def __len__(self):
        return self.root.size

    # -- rotations (maintain subtree sizes) ---------------------------------
    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left is not self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self.nil:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right is not self.nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is self.nil:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    # -- search -------------------------------------------------------------
    def _find(self, key):
        x = self.root
        while x is not self.nil:
            if key == x.key:
                return x
            x = x.left if key < x.key else x.right
        return self.nil

    def search(self, key):
        node = self._find(key)
        return None if node is self.nil else node.value

    def __contains__(self, key):
        return self._find(key) is not self.nil

    # -- insert -------------------------------------------------------------
    def insert(self, key, value=None):
        # overwrite if present
        existing = self._find(key)
        if existing is not self.nil:
            existing.value = value
            return
        z = _Node(key, value, RED, self.nil)
        z.left = z.right = self.nil
        y = self.nil
        x = self.root
        while x is not self.nil:
            y = x
            x.size += 1                        # every ancestor gains a descendant
            x = x.left if key < x.key else x.right
        z.parent = y
        if y is self.nil:
            self.root = z
        elif key < y.key:
            y.left = z
        else:
            y.right = z
        self._insert_fixup(z)

    def __setitem__(self, key, value):
        self.insert(key, value)

    def _insert_fixup(self, z):
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(z.parent.parent)
        self.root.color = BLACK

    # -- delete -------------------------------------------------------------
    def _transplant(self, u, v):
        if u.parent is self.nil:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, x):
        while x.left is not self.nil:
            x = x.left
        return x

    def delete(self, key):
        z = self._find(key)
        if z is self.nil:
            return False
        # decrement sizes along the path from the root to z
        p = z.parent
        cur = z
        # walk up decrementing (do after we know z exists)
        node = z
        while node is not self.nil:
            node.size -= 1
            node = node.parent

        y = z
        y_original_color = y.color
        if z.left is self.nil:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is self.nil:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            # y will move into z's place; fix sizes on the path from y up to z's right subtree
            # decrement along y's original ancestors down to (not including) z
            n = y.parent
            while n is not z and n is not self.nil:
                n.size -= 1
                n = n.parent
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
            y.size = y.left.size + y.right.size + 1
        if y_original_color == BLACK:
            self._delete_fixup(x)
        return True

    def _delete_fixup(self, x):
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        x.color = BLACK

    # -- ordered access -----------------------------------------------------
    def items(self):
        out = []
        self._inorder(self.root, out)
        return out

    def keys(self):
        return [k for k, _ in self.items()]

    def _inorder(self, node, out):
        if node is self.nil:
            return
        self._inorder(node.left, out)
        out.append((node.key, node.value))
        self._inorder(node.right, out)

    def min_key(self):
        if self.root is self.nil:
            return None
        return self._minimum(self.root).key

    def max_key(self):
        if self.root is self.nil:
            return None
        x = self.root
        while x.right is not self.nil:
            x = x.right
        return x.key

    # -- order statistics ---------------------------------------------------
    def select(self, k):
        """The k-th smallest key (0-indexed). Raises IndexError if out of range."""
        if k < 0 or k >= self.root.size:
            raise IndexError("select index out of range")
        x = self.root
        while x is not self.nil:
            left_size = x.left.size
            if k == left_size:
                return x.key
            elif k < left_size:
                x = x.left
            else:
                k -= left_size + 1
                x = x.right
        raise IndexError("select failed")      # unreachable

    def rank(self, key):
        """Number of keys strictly less than ``key`` (its 0-indexed position if present)."""
        r = 0
        x = self.root
        while x is not self.nil:
            if key <= x.key:
                x = x.left
            else:
                r += x.left.size + 1
                x = x.right
        return r

    # -- invariant checking (for validation) --------------------------------
    def check_invariants(self):
        """Verify the red-black invariants; returns True if all hold."""
        if self.root is self.nil:
            return True
        if self.root.color != BLACK:
            return False
        return self._check(self.root)[0]

    def _check(self, node):
        """Returns (ok, black_height) for the subtree rooted at node."""
        if node is self.nil:
            return True, 1
        # no red node has a red child
        if node.color == RED:
            if node.left.color == RED or node.right.color == RED:
                return False, 0
        # subtree size correct
        if node.size != node.left.size + node.right.size + 1:
            return False, 0
        # BST order
        if node.left is not self.nil and node.left.key >= node.key:
            return False, 0
        if node.right is not self.nil and node.right.key <= node.key:
            return False, 0
        lok, lbh = self._check(node.left)
        rok, rbh = self._check(node.right)
        if not lok or not rok or lbh != rbh:
            return False, 0
        return True, lbh + (1 if node.color == BLACK else 0)

    def height(self):
        return self._height(self.root)

    def _height(self, node):
        if node is self.nil:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))
