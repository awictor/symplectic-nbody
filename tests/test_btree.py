"""Tests for btree: mirror dict + sorted over random ops, invariants hold, range queries, shallow."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from btree import BTree  # noqa: E402


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
    # ---- 1. basic insert / search / order --------------------------------------------
    bt = BTree(t=3)
    for k in [10, 20, 5, 6, 12, 30, 7, 17]:
        bt.insert(k, k * 100)
    check("search finds inserted keys", all(bt.search(k) == k * 100 for k in [10, 20, 5, 6, 12, 30, 7, 17]))
    check("search misses absent key", bt.search(99) is None)
    check("in-order traversal is sorted", bt.keys() == sorted([10, 20, 5, 6, 12, 30, 7, 17]))
    check("invariants hold after inserts", bt.check_invariants())
    check("min/max keys", bt.min_key() == 5 and bt.max_key() == 30)

    # ---- 2. mirror a dict over thousands of random ops, several degrees ---------------
    for t in (2, 3, 5, 8):
        rng = LCG(1000 + t)
        bt = BTree(t=t)
        ref = {}
        inv_ok = True
        agree = True
        for step in range(3000):
            op = rng.randint(0, 3)
            key = rng.randint(0, 200)
            if op <= 1:                       # insert / overwrite (weighted)
                val = rng.nxt()
                bt.insert(key, val)
                ref[key] = val
            elif op == 2:                     # delete
                res = bt.delete(key)
                had = key in ref
                if res != had:
                    agree = False
                ref.pop(key, None)
            else:                             # lookup + membership
                if bt.search(key) != ref.get(key):
                    agree = False
                if (key in bt) != (key in ref):
                    agree = False
            if not bt.check_invariants():
                inv_ok = False
                break
        check(f"t={t}: agrees with dict over 3000 ops", agree)
        check(f"t={t}: invariants hold throughout", inv_ok)
        check(f"t={t}: in-order equals sorted dict keys", bt.keys() == sorted(ref.keys()))
        check(f"t={t}: size matches dict", len(bt) == len(ref))

    # ---- 3. overwrite doesn't grow size ----------------------------------------------
    bt = BTree(t=4)
    bt.insert(5, "a")
    bt.insert(5, "b")
    check("overwrite updates value", bt.search(5) == "b")
    check("overwrite keeps size 1", len(bt) == 1)

    # ---- 4. range queries match the sorted slice -------------------------------------
    rng = LCG(77)
    bt = BTree(t=3)
    keys = list({rng.randint(0, 1000) for _ in range(400)})
    for k in keys:
        bt.insert(k, k)
    skeys = sorted(keys)
    range_bad = 0
    for _ in range(100):
        lo = rng.randint(0, 1000)
        hi = rng.randint(lo, 1000)
        got = [k for k, _ in bt.range(lo, hi)]
        expected = [k for k in skeys if lo <= k <= hi]
        if got != expected:
            range_bad += 1
    check("range queries match sorted slice", range_bad == 0, f"{range_bad} failures")

    # ---- 5. tree stays shallow (height ~ log_t n) ------------------------------------
    n = len(keys)
    t = 3
    max_h = 1 + math.log(max(2, n), t)   # loose upper bound
    check("tree height is logarithmic", bt.height() <= max_h + 2,
          f"height {bt.height()} for n={n} (log_t n = {math.log(n, t):.1f})")

    # ---- 6. delete everything empties the tree ---------------------------------------
    rng = LCG(555)
    bt = BTree(t=4)
    keys = list({rng.randint(0, 5000) for _ in range(600)})
    for k in keys:
        bt.insert(k, k)
    order = keys[:]
    # shuffle deletion order
    for i in range(len(order) - 1, 0, -1):
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]
    empty_ok = True
    for k in order:
        bt.delete(k)
        if not bt.check_invariants():
            empty_ok = False
            break
    check("delete every key keeps invariants", empty_ok)
    check("tree is empty after deleting all", len(bt) == 0 and bt.keys() == [])
    check("empty tree search returns None", bt.search(1) is None)

    # ---- 7. degree validation --------------------------------------------------------
    try:
        BTree(t=1)
        check("t<2 rejected", False)
    except ValueError:
        check("t<2 rejected", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
