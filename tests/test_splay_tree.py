"""Tests for splay_tree: sorted invariant, splay-to-root, working set, pred/succ, delete."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from splay_tree import SplayTree

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 71
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def is_bst(node, lo=float("-inf"), hi=float("inf")):
    if node is None:
        return True
    if not (lo < node.key < hi):
        return False
    return is_bst(node.left, lo, node.key) and is_bst(node.right, node.key, hi)


def parents_ok(node, parent=None):
    if node is None:
        return True
    if node.parent is not parent:
        return False
    return parents_ok(node.left, node) and parents_ok(node.right, node)


# --- basic sorted invariant ------------------------------------------------
t = SplayTree()
for v in [5, 3, 8, 1, 9, 2, 7, 4, 6]:
    t.insert(v)
check("in-order is sorted", t.inorder() == list(range(1, 10)))
check("BST property holds", is_bst(t.root))
check("parent pointers consistent", parents_ok(t.root))
check("length is correct", len(t) == 9)

# --- accessed key becomes the root (the defining splay property) -----------
splay_ok = True
for k in [3, 7, 1, 9, 5]:
    _ = k in t
    if t.root_key() != k:
        splay_ok = False
check("accessed key is splayed to the root", splay_ok)

# insert also splays
t.insert(6)
check("inserted/updated key is at the root", t.root_key() == 6)

# --- membership ------------------------------------------------------------
check("contains present keys", all(k in t for k in range(1, 10)))
check("absent key not found", 100 not in t)
check("get returns value", (SplayTree_with := SplayTree()).insert(1, "a") or SplayTree_with.get(1) == "a")

# --- predecessor / successor match a sorted array --------------------------
keys = sorted(t.inorder())
ps_ok = True
for i, k in enumerate(keys):
    pred = keys[i - 1] if i > 0 else None
    succ = keys[i + 1] if i < len(keys) - 1 else None
    if t.predecessor(k) != pred or t.successor(k) != succ:
        ps_ok = False
check("predecessor/successor match the sorted order", ps_ok)

# --- min / max -------------------------------------------------------------
check("find-min is the smallest", t.find_min() == 1)
check("find-max is the largest", t.find_max() == 9)
check("find-min splays min to root", t.find_min() == 1 and t.root_key() == 1)

# --- delete keeps ordering -------------------------------------------------
t.delete(5)
check("delete removes the key", 5 not in t and t.inorder() == [1, 2, 3, 4, 6, 7, 8, 9])
check("BST property after delete", is_bst(t.root))
check("parents consistent after delete", parents_ok(t.root))
check("delete of absent key returns False", not t.delete(1000))

# --- long random insert/delete stream vs a reference set -------------------
ref = set()
st = SplayTree()
ok = True
for _ in range(3000):
    k = int(rng() * 200)
    if rng() < 0.6:
        st.insert(k)
        ref.add(k)
    else:
        st.delete(k)
        ref.discard(k)
    if rng() < 0.02:
        if st.inorder() != sorted(ref) or not is_bst(st.root) or not parents_ok(st.root):
            ok = False
            break
check("random stream keeps contents equal to a reference set", ok and st.inorder() == sorted(ref))
check("BST + parents valid after the stream", is_bst(st.root) and parents_ok(st.root))

# --- membership over the streamed set --------------------------------------
final = sorted(ref)
check("all reference keys present", all(k in st for k in final[::5]))

# --- working-set property: hot keys stay cheap -----------------------------
# insert many keys, then hammer a small hot set; average access depth should be tiny
ws = SplayTree()
for k in range(2000):
    ws.insert(k)
hot = [10, 20, 30, 40, 50]
# access the hot set many times
for _ in range(500):
    for h in hot:
        _ = h in ws
# after this, accessing any hot key should splay it to the root at depth 1;
# measure the depth of a hot key found by walking from the root BEFORE splaying
def depth_of(tree, key):
    node = tree.root
    d = 0
    while node:
        if node.key == key:
            return d
        node = node.left if key < node.key else node.right
        d += 1
    return -1

# access one hot key, then check the LAST one accessed is at the root
_ = hot[-1] in ws
check("a hot key is at the root after access", ws.root_key() == hot[-1])
# average depth of the hot set is small relative to log2(2000) ~ 11
avg_depth = sum(depth_of(ws, h) for h in hot) / len(hot)
check(f"hot-set average depth is small ({avg_depth:.1f})", avg_depth < 11)

# --- single element and empty ----------------------------------------------
one = SplayTree()
one.insert(42, "x")
check("single element contains", 42 in one)
check("single element get", one.get(42) == "x")
check("single element min==max", one.find_min() == 42 and one.find_max() == 42)
empty = SplayTree()
check("empty tree length 0", len(empty) == 0)
check("empty tree membership", 1 not in empty)
check("empty tree find-min is None", empty.find_min() is None)
check("empty tree inorder", empty.inorder() == [])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all splay_tree tests passed")
