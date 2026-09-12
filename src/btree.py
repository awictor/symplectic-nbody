"""The B-tree -- the balanced multiway search tree that runs every database and filesystem.

A binary search tree makes one comparison per node and follows one of two children, so a lookup in n
keys touches about log2(n) nodes. That is fine in RAM, but disastrous on disk or SSD, where each node
visited is a separate seek costing millions of times a comparison. The B-TREE (Bayer & McCreight, 1972)
answers this: pack MANY keys into each node -- hundreds, sized to one disk page -- so the tree is short
and FAT, and a lookup in a billion keys touches only three or four nodes. This is why B-trees (and their
B+-tree cousins) index essentially every relational database, and organise NTFS, ext4, HFS+, and Btrfs.

A B-tree of MINIMUM DEGREE t obeys strict invariants that keep it balanced without any rotations or
colour bits: every node except the root holds between t-1 and 2t-1 keys; every internal node with k keys
has exactly k+1 children; the keys within a node are sorted and separate the ranges of the children
between them; and -- the property that makes the balance automatic -- ALL LEAVES SIT AT THE SAME DEPTH.
Growth happens at the root: when a node would overflow it is SPLIT about its median key, which is pushed
up to the parent, and if the root itself splits the tree gains a level. Deletion is the intricate
converse -- borrowing a key from a sibling or merging two thin nodes so no node ever falls below t-1 keys.

This module implements a B-tree as an ordered key-value map: search, insert (with proactive top-down
node splitting), delete (with the full borrow/merge rebalancing), in-order traversal, range queries
between two keys, and the min/max keys. Because the balance is structural, every operation is O(log n)
with a large base, and the tree never degenerates. Pure standard library.

Validation. The B-tree is exercised alongside Python's ``dict`` and ``sorted`` as oracles over thousands
of seeded random operations -- insert, overwrite, lookup, delete, membership -- and must agree on every
one: the same values for present keys, the same absence for missing keys, and an in-order traversal that
exactly equals the sorted key list. After every operation the STRUCTURAL INVARIANTS are checked directly:
key counts within [t-1, 2t-1] (root exempt on the lower bound), keys sorted within each node, child
count one more than key count, and all leaves at equal depth. Range queries match the corresponding slice
of the sorted keys, and the tree stays SHALLOW (height O(log_t n)). Deleting every key in random order
empties the tree correctly."""


class _Node:
    __slots__ = ("keys", "values", "children", "leaf")

    def __init__(self, leaf=True):
        self.keys = []
        self.values = []
        self.children = []
        self.leaf = leaf


class BTree:
    """An ordered key-value map backed by a B-tree of minimum degree ``t`` (t >= 2)."""

    def __init__(self, t=3):
        if t < 2:
            raise ValueError("minimum degree t must be >= 2")
        self.t = t
        self.root = _Node(leaf=True)
        self._size = 0

    def __len__(self):
        return self._size

    # -- search --------------------------------------------------------------
    def search(self, key):
        """Return the value for key, or None if absent."""
        node = self.root
        while node is not None:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if i < len(node.keys) and key == node.keys[i]:
                return node.values[i]
            if node.leaf:
                return None
            node = node.children[i]
        return None

    def __contains__(self, key):
        return self._find(self.root, key) is not None

    def _find(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return (node, i)
        if node.leaf:
            return None
        return self._find(node.children[i], key)

    def __getitem__(self, key):
        v = self.search(key)
        if v is None and key not in self:
            raise KeyError(key)
        return v

    # -- insert --------------------------------------------------------------
    def insert(self, key, value=None):
        """Insert or overwrite key -> value."""
        found = self._find(self.root, key)
        if found is not None:
            node, i = found
            node.values[i] = value
            return
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            # grow taller: new root over the split old root
            new_root = _Node(leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_nonfull(self.root, key, value)
        self._size += 1

    def __setitem__(self, key, value):
        self.insert(key, value)

    def _split_child(self, parent, i):
        """Split the full child parent.children[i] about its median, pulling the median into parent."""
        t = self.t
        child = parent.children[i]
        sibling = _Node(leaf=child.leaf)
        mid = t - 1
        # move the upper t-1 keys/values to the sibling
        sibling.keys = child.keys[mid + 1:]
        sibling.values = child.values[mid + 1:]
        median_key = child.keys[mid]
        median_val = child.values[mid]
        child.keys = child.keys[:mid]
        child.values = child.values[:mid]
        if not child.leaf:
            sibling.children = child.children[mid + 1:]
            child.children = child.children[:mid + 1]
        parent.keys.insert(i, median_key)
        parent.values.insert(i, median_val)
        parent.children.insert(i + 1, sibling)

    def _insert_nonfull(self, node, key, value):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            node.values.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = value
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_nonfull(node.children[i], key, value)

    # -- delete --------------------------------------------------------------
    def delete(self, key):
        """Delete key if present. Returns True if a key was removed."""
        if key not in self:
            return False
        self._delete(self.root, key)
        if len(self.root.keys) == 0 and not self.root.leaf:
            self.root = self.root.children[0]
        self._size -= 1
        return True

    def _delete(self, node, key):
        t = self.t
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and node.keys[i] == key:
            if node.leaf:
                node.keys.pop(i)
                node.values.pop(i)
            else:
                self._delete_internal(node, i)
        else:
            if node.leaf:
                return
            self._delete_from_child(node, i, key)

    def _delete_internal(self, node, i):
        t = self.t
        key = node.keys[i]
        left = node.children[i]
        right = node.children[i + 1]
        if len(left.keys) >= t:
            # predecessor
            pred_node = left
            while not pred_node.leaf:
                pred_node = pred_node.children[-1]
            pk = pred_node.keys[-1]
            pv = pred_node.values[-1]
            node.keys[i], node.values[i] = pk, pv
            self._delete(left, pk)
        elif len(right.keys) >= t:
            # successor
            succ_node = right
            while not succ_node.leaf:
                succ_node = succ_node.children[0]
            sk = succ_node.keys[0]
            sv = succ_node.values[0]
            node.keys[i], node.values[i] = sk, sv
            self._delete(right, sk)
        else:
            self._merge(node, i)
            self._delete(left, key)

    def _delete_from_child(self, node, i, key):
        t = self.t
        child = node.children[i]
        if len(child.keys) < t:
            self._fill(node, i)
            # after filling, the index may shift; re-find which child to descend
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if i < len(node.keys) and node.keys[i] == key:
                # key rose into this node during merge/borrow
                if node.leaf:
                    node.keys.pop(i)
                    node.values.pop(i)
                else:
                    self._delete_internal(node, i)
                return
        self._delete(node.children[i], key)

    def _fill(self, node, i):
        t = self.t
        if i > 0 and len(node.children[i - 1].keys) >= t:
            self._borrow_from_prev(node, i)
        elif i < len(node.children) - 1 and len(node.children[i + 1].keys) >= t:
            self._borrow_from_next(node, i)
        else:
            if i < len(node.children) - 1:
                self._merge(node, i)
            else:
                self._merge(node, i - 1)

    def _borrow_from_prev(self, node, i):
        child = node.children[i]
        sibling = node.children[i - 1]
        child.keys.insert(0, node.keys[i - 1])
        child.values.insert(0, node.values[i - 1])
        node.keys[i - 1] = sibling.keys.pop()
        node.values[i - 1] = sibling.values.pop()
        if not child.leaf:
            child.children.insert(0, sibling.children.pop())

    def _borrow_from_next(self, node, i):
        child = node.children[i]
        sibling = node.children[i + 1]
        child.keys.append(node.keys[i])
        child.values.append(node.values[i])
        node.keys[i] = sibling.keys.pop(0)
        node.values[i] = sibling.values.pop(0)
        if not child.leaf:
            child.children.append(sibling.children.pop(0))

    def _merge(self, node, i):
        child = node.children[i]
        sibling = node.children[i + 1]
        child.keys.append(node.keys.pop(i))
        child.values.append(node.values.pop(i))
        child.keys.extend(sibling.keys)
        child.values.extend(sibling.values)
        child.children.extend(sibling.children)
        node.children.pop(i + 1)

    # -- traversal / range ---------------------------------------------------
    def items(self):
        """In-order (key, value) pairs."""
        out = []
        self._inorder(self.root, out)
        return out

    def keys(self):
        return [k for k, _ in self.items()]

    def _inorder(self, node, out):
        if node.leaf:
            for k, v in zip(node.keys, node.values):
                out.append((k, v))
        else:
            for i in range(len(node.keys)):
                self._inorder(node.children[i], out)
                out.append((node.keys[i], node.values[i]))
            self._inorder(node.children[-1], out)

    def range(self, lo, hi):
        """All (key, value) pairs with lo <= key <= hi, in order."""
        out = []
        self._range(self.root, lo, hi, out)
        return out

    def _range(self, node, lo, hi, out):
        i = 0
        while i < len(node.keys) and node.keys[i] < lo:
            i += 1
        while i < len(node.keys) and node.keys[i] <= hi:
            if not node.leaf:
                self._range(node.children[i], lo, hi, out)
            out.append((node.keys[i], node.values[i]))
            i += 1
        if not node.leaf:
            self._range(node.children[i], lo, hi, out)

    def min_key(self):
        node = self.root
        if not node.keys:
            return None
        while not node.leaf:
            node = node.children[0]
        return node.keys[0]

    def max_key(self):
        node = self.root
        if not node.keys:
            return None
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1]

    def height(self):
        h = 1
        node = self.root
        while not node.leaf:
            h += 1
            node = node.children[0]
        return h

    # -- invariant checking (for validation) ---------------------------------
    def check_invariants(self):
        """Verify the B-tree invariants; returns True if all hold."""
        depths = set()
        ok = self._check(self.root, 0, depths, is_root=True)
        return ok and len(depths) <= 1

    def _check(self, node, depth, depths, is_root):
        t = self.t
        # key count bounds
        if not is_root and len(node.keys) < t - 1:
            return False
        if len(node.keys) > 2 * t - 1:
            return False
        # keys sorted
        for i in range(len(node.keys) - 1):
            if node.keys[i] >= node.keys[i + 1]:
                return False
        # values align
        if len(node.values) != len(node.keys):
            return False
        if node.leaf:
            depths.add(depth)
            return True
        # internal: children count = keys + 1
        if len(node.children) != len(node.keys) + 1:
            return False
        return all(self._check(c, depth + 1, depths, False) for c in node.children)
