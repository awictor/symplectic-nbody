"""Tests for treap: sorted invariant, heap property, order statistics, split/merge, balance."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from treap import Treap, build, _size

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 123
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- in-order traversal is sorted ------------------------------------------
t = build([5, 3, 8, 1, 9, 2, 7, 4, 6])
check("in-order is sorted", t.inorder() == [1, 2, 3, 4, 5, 6, 7, 8, 9])
check("length is correct", len(t) == 9)

# --- heap property holds on priorities -------------------------------------
def heap_ok(node):
    if node is None:
        return True
    for c in (node.left, node.right):
        if c is not None and c.priority > node.priority:
            return False
    return heap_ok(node.left) and heap_ok(node.right)
check("max-heap property on priorities", heap_ok(t.root))

# --- subtree sizes are consistent ------------------------------------------
def size_ok(node):
    if node is None:
        return True
    expected = 1 + _size(node.left) + _size(node.right)
    if node.size != expected:
        return False
    return size_ok(node.left) and size_ok(node.right)
check("subtree size augmentation is consistent", size_ok(t.root))

# --- membership ------------------------------------------------------------
check("contains present keys", all(k in t for k in [1, 5, 9]))
check("absent key not contained", 100 not in t)
check("duplicate insert is a no-op", not t.insert(5) and len(t) == 9)

# --- select and rank match a sorted array ----------------------------------
arr = sorted(t.inorder())
check("select matches sorted array", all(t.select(k) == arr[k] for k in range(len(arr))))
check("rank matches array position", all(t.rank(arr[k]) == k for k in range(len(arr))))
check("rank of a value between keys", t.rank(5) == arr.index(5))

# --- delete ----------------------------------------------------------------
t.delete(5)
check("delete removes the key", 5 not in t and t.inorder() == [1, 2, 3, 4, 6, 7, 8, 9])
check("delete of absent key returns False", not t.delete(1000))
check("heap property after delete", heap_ok(t.root))
check("sizes consistent after delete", size_ok(t.root))

# --- split produces the correct partition ----------------------------------
t2 = build(list(range(20)), seed=7)
left, right = t2._split(t2.root, 10)
def collect(node, out):
    if node:
        collect(node.left, out); out.append(node.key); collect(node.right, out)
lo, hi = [], []
collect(left, lo); collect(right, hi)
check("split lower part is keys < 10", sorted(lo) == list(range(10)))
check("split upper part is keys >= 10", sorted(hi) == list(range(10, 20)))
# merge them back
t2.root = t2._merge(left, right)
check("merge reassembles the original set", t2.inorder() == list(range(20)))

# --- long random stream vs a reference set ---------------------------------
import random
ref = set()
tr = Treap(seed=42)
ok = True
for _ in range(3000):
    k = int(rng() * 200)
    if rng() < 0.6:
        tr.insert(k)
        ref.add(k)
    else:
        tr.delete(k)
        ref.discard(k)
    # occasional full check
    if rng() < 0.02:
        if tr.inorder() != sorted(ref):
            ok = False
            break
check("random insert/delete stream matches a reference set", ok and tr.inorder() == sorted(ref))
check("heap property holds after the stream", heap_ok(tr.root))
check("sizes consistent after the stream", size_ok(tr.root))

# --- order statistics over the streamed set --------------------------------
final = sorted(ref)
check("select over streamed set", all(tr.select(k) == final[k] for k in range(0, len(final), 7)))
check("rank over streamed set", all(tr.rank(final[k]) == k for k in range(0, len(final), 7)))

# --- balance: height within a small constant of 2 log2 n -------------------
big = Treap(seed=5)
for _ in range(10000):
    big.insert(int(rng() * 10 ** 9))
n = len(big)
expected = 2 * math.log2(n)
check(f"height {big.height()} within ~3x the 2log2(n)={expected:.1f} expectation",
      big.height() < 3 * expected)

# --- select out of range raises --------------------------------------------
raised = False
try:
    build([1, 2, 3]).select(5)
except IndexError:
    raised = True
check("select out of range raises IndexError", raised)

# --- empty treap -----------------------------------------------------------
empty = Treap()
check("empty treap length 0", len(empty) == 0)
check("empty treap inorder", empty.inorder() == [])
check("empty treap membership", 1 not in empty)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all treap tests passed")
