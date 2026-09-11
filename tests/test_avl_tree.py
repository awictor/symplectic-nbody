"""Tests for avl_tree: balance invariant, log height, brute-force agreement, rotations, ranges."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from avl_tree import AVLTree

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- basic ops -------------------------------------------------------------
t = AVLTree()
for k in [5, 2, 8, 1, 9, 3, 7]:
    t.insert(k, k * 10)
check("keys in sorted order", t.keys() == [1, 2, 3, 5, 7, 8, 9])
check("search returns value", t.search(8) == 80)
check("search missing returns default", t.search(4) is None)
check("contains", 3 in t and 4 not in t)
check("length", len(t) == 7)
check("min/max", t.min() == 1 and t.max() == 9)
check("items sorted with values", t.items() == [(1, 10), (2, 20), (3, 30), (5, 50), (7, 70), (8, 80), (9, 90)])
check("stays balanced", t.is_balanced())
check("is a BST", t.is_bst())

# --- update, not duplicate -------------------------------------------------
added = t.insert(5, 555)
check("re-insert updates value", t.search(5) == 555)
check("re-insert returns False", added is False)
check("re-insert keeps length", len(t) == 7)

# --- sorted insertion keeps the height logarithmic -------------------------
srt = AVLTree()
for k in range(1, 128):
    srt.insert(k, k)
check("127 sorted inserts stay balanced", srt.is_balanced())
check("height is logarithmic, not linear", srt.height() <= 2 * math.log2(128) + 1)
check("sorted insert preserves order", srt.keys() == list(range(1, 128)))

# --- all four rotation cases fix the imbalance -----------------------------
for seq, name in [([3, 2, 1], "LL"), ([1, 2, 3], "RR"), ([3, 1, 2], "LR"), ([1, 3, 2], "RL")]:
    tt = AVLTree()
    for k in seq:
        tt.insert(k)
    check(f"{name} rotation balances (root=2)", tt.root.key == 2 and tt.is_balanced())

# --- deletion --------------------------------------------------------------
check("delete present key", t.delete(8))
check("deleted key gone", 8 not in t and t.keys() == [1, 2, 3, 5, 7, 9])
check("length drops", len(t) == 6)
check("balanced after delete", t.is_balanced())
check("delete absent returns False", not t.delete(42))

# --- delete a node with two children (successor replacement) ---------------
t2 = AVLTree()
for k in [10, 5, 15, 3, 7, 12, 20]:
    t2.insert(k, k)
check("delete two-child node", t2.delete(10) and t2.is_bst() and t2.is_balanced())
check("remaining keys correct", t2.keys() == [3, 5, 7, 12, 15, 20])

# --- range queries ---------------------------------------------------------
check("range returns in-range keys", [k for k, _ in t.range(2, 7)] == [2, 3, 5, 7])
check("range excludes out-of-range", all(2 <= k <= 7 for k, _ in t.range(2, 7)))
check("empty range", t.range(100, 200) == [])
check("range over everything", len(t.range(-10, 100)) == len(t))

# --- empty tree edge cases -------------------------------------------------
empty = AVLTree()
check("empty length 0", len(empty) == 0)
check("empty keys", empty.keys() == [])
check("empty height 0", empty.height() == 0)


def raises_keyerror(fn):
    try:
        fn()
        return False
    except KeyError:
        return True


check("min of empty raises", raises_keyerror(empty.min))
check("max of empty raises", raises_keyerror(empty.max))

# --- brute-force agreement + invariant maintained throughout ---------------
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


avl = AVLTree()
ref = {}
always_balanced = True
for _ in range(5000):
    k = int(rng() * 200)
    if rng() < 0.5:
        avl.insert(k, k * 3)
        ref[k] = k * 3
    else:
        avl.delete(k)
        ref.pop(k, None)
    if not avl.is_balanced():
        always_balanced = False
        break
check("balance invariant held through 5000 ops", always_balanced)
check("keys match a reference dict", avl.keys() == sorted(ref))
check("length matches reference", len(avl) == len(ref))
check("every search matches reference", all(avl.search(k) == ref.get(k) for k in range(200)))
check("still a valid BST", avl.is_bst())
check("height stays logarithmic", avl.height() <= 2 * math.log2(max(2, len(avl))) + 2)
check("random range matches reference",
      avl.range(50, 120) == sorted((k, v) for k, v in ref.items() if 50 <= k <= 120))

# --- string keys -----------------------------------------------------------
words = AVLTree()
for w in ["delta", "alpha", "charlie", "bravo"]:
    words.insert(w, len(w))
check("string keys sort lexicographically", words.keys() == ["alpha", "bravo", "charlie", "delta"])
check("string tree balanced", words.is_balanced())

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all avl_tree tests passed")
