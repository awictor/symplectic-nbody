"""Tests for red_black_tree: mirrors dict + sorted, RB invariants, order statistics, height bound."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from red_black_tree import RedBlackTree  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def nxt(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        return lo + (self.nxt() >> 8) % (hi - lo + 1)


def main():
    # ---- 1. basic operations ----------------------------------------------------------
    t = RedBlackTree()
    for k in [5, 3, 8, 1, 4, 7, 9, 2, 6]:
        t.insert(k, k * 10)
    check("search finds inserted", all(t.search(k) == k * 10 for k in range(1, 10)))
    check("in-order sorted", t.keys() == list(range(1, 10)))
    check("invariants after inserts", t.check_invariants())
    check("min/max", t.min_key() == 1 and t.max_key() == 9)
    check("len correct", len(t) == 9)
    check("absent key None", t.search(100) is None)

    # ---- 2. mirror a dict over thousands of random ops --------------------------------
    rng = LCG(2024)
    t = RedBlackTree()
    ref = {}
    agree = True
    inv_ok = True
    order_ok = True
    for step in range(6000):
        op = rng.randint(0, 3)
        k = rng.randint(0, 300)
        if op <= 1:
            v = rng.nxt()
            t.insert(k, v)
            ref[k] = v
        elif op == 2:
            had = k in ref
            r = t.delete(k)
            if r != had:
                agree = False
            ref.pop(k, None)
        else:
            if t.search(k) != ref.get(k):
                agree = False
            if (k in t) != (k in ref):
                agree = False
        if not t.check_invariants():
            inv_ok = False
            break
        if t.keys() != sorted(ref):
            order_ok = False
            break
    check("agrees with dict over 6000 ops", agree)
    check("RB invariants hold throughout", inv_ok)
    check("in-order equals sorted keys throughout", order_ok)
    check("size matches dict", len(t) == len(ref))

    # ---- 3. order statistics vs the sorted key list -----------------------------------
    sk = sorted(ref)
    sel_ok = all(t.select(i) == sk[i] for i in range(len(sk)))
    check("select(k) equals k-th sorted key", sel_ok)
    rank_ok = True
    for _ in range(200):
        k = rng.randint(0, 300)
        expected = sum(1 for x in sk if x < k)
        if t.rank(k) != expected:
            rank_ok = False
    check("rank(x) equals number of keys < x", rank_ok)
    # select out of range raises
    try:
        t.select(len(t))
        check("select out of range raises", False)
    except IndexError:
        check("select out of range raises", True)

    # ---- 4. height stays within 2 log2(n+1) -------------------------------------------
    check("height within red-black bound", t.height() <= 2 * math.log2(len(t) + 1) + 1e-9,
          f"height {t.height()} for n={len(t)} (bound {2*math.log2(len(t)+1):.1f})")

    # ---- 5. sorted-input insertion stays balanced (would ruin a plain BST) ------------
    t = RedBlackTree()
    for k in range(1000):
        t.insert(k, k)
    check("sorted insertion stays balanced", t.check_invariants() and
          t.height() <= 2 * math.log2(1001) + 1e-9, f"height {t.height()}")
    check("sorted insertion in order", t.keys() == list(range(1000)))

    # ---- 6. overwrite doesn't grow size -----------------------------------------------
    t = RedBlackTree()
    t.insert(5, "a")
    t.insert(5, "b")
    check("overwrite updates value", t.search(5) == "b")
    check("overwrite keeps size 1", len(t) == 1)

    # ---- 7. delete everything empties the tree ----------------------------------------
    rng = LCG(99)
    t = RedBlackTree()
    keys = list({rng.randint(0, 5000) for _ in range(800)})
    for k in keys:
        t.insert(k, k)
    order = keys[:]
    for i in range(len(order) - 1, 0, -1):
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]
    empty_ok = True
    for k in order:
        t.delete(k)
        if not t.check_invariants():
            empty_ok = False
            break
    check("delete every key keeps invariants", empty_ok)
    check("tree empty after deleting all", len(t) == 0 and t.keys() == [])

    # ---- 8. delete non-existent key is a no-op ----------------------------------------
    t = RedBlackTree()
    t.insert(1, 1)
    check("delete missing key returns False", t.delete(99) is False and len(t) == 1)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
