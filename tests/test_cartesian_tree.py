"""Tests for Cartesian tree: inorder=sequence, heap property, O(n)==naive, RMQ via LCA == sparse-table."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cartesian_tree import (  # noqa: E402
    build_cartesian_tree,
    inorder,
    is_heap_ordered,
    CartesianRMQ,
    build_naive,
    brute_min_index,
)
from sparse_table import RMQ  # noqa: E402


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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _distinct_perm(n, rng):
    """A random permutation of 0..n-1 (distinct values)."""
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def main():
    rng = _lcg(1)

    # ---- 1. in-order traversal reproduces the sequence positions ------------------------
    ok = True
    for _ in range(30):
        n = 1 + int(rng() * 30)
        a = _distinct_perm(n, rng)
        root, left, right, parent = build_cartesian_tree(a)
        if inorder(root, left, right) != list(range(n)):
            ok = False
            break
    check("in-order = 0..n-1 (BST on position)", ok)

    # ---- 2. heap property holds at every node -------------------------------------------
    ok = True
    for _ in range(30):
        n = 1 + int(rng() * 30)
        a = _distinct_perm(n, rng)
        root, left, right, parent = build_cartesian_tree(a)
        if not is_heap_ordered(a, left, right, parent):
            ok = False
            break
    check("min-heap property holds", ok)

    # ---- 3. root is the global minimum --------------------------------------------------
    ok = True
    for _ in range(20):
        n = 2 + int(rng() * 20)
        a = _distinct_perm(n, rng)
        root, left, right, parent = build_cartesian_tree(a)
        if a[root] != min(a):
            ok = False
            break
    check("root is the global minimum", ok)

    # ---- 4. O(n) stack build == naive recursive build -----------------------------------
    ok = True
    for _ in range(30):
        n = 1 + int(rng() * 25)
        a = _distinct_perm(n, rng)
        fast = build_cartesian_tree(a)
        slow = build_naive(a)
        # compare left/right/parent arrays (roots too)
        if fast[0] != slow[0] or fast[1] != slow[1] or fast[2] != slow[2] or fast[3] != slow[3]:
            ok = False
            check("O(n) build == naive", False, f"a={a}")
            break
    if ok:
        check("O(n) stack build == naive recursive build", True)

    # ---- 5. RMQ via tree-LCA == sparse-table RMQ over every subrange --------------------
    ok = True
    for _ in range(15):
        n = 2 + int(rng() * 25)
        a = _distinct_perm(n, rng)
        crmq = CartesianRMQ(a)
        st = RMQ(a)
        for l in range(n):
            for r in range(l, n):
                cv = crmq.min_value(l, r)          # inclusive [l, r]
                sv = st.min_range(l, r + 1)        # sparse-table is half-open [l, r)
                if cv != sv:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            check("Cartesian RMQ == sparse-table RMQ", False, f"a={a}")
            break
    if ok:
        check("Cartesian-tree RMQ == sparse-table RMQ (all subranges)", True)

    # ---- 6. RMQ index matches brute force -----------------------------------------------
    a = [5, 2, 8, 1, 9, 3, 7, 4]
    crmq = CartesianRMQ(a)
    ok = True
    for l in range(len(a)):
        for r in range(l, len(a)):
            if crmq.min_index(l, r) != brute_min_index(a, l, r):
                ok = False
                break
    check("min_index == brute force", ok)

    # ---- 7. specific known tree for [3,1,2] ---------------------------------------------
    # min is index 1 (value 1) -> root; left subtree {0}, right subtree {2}
    root, left, right, parent = build_cartesian_tree([3, 1, 2])
    check("root of [3,1,2] is index 1", root == 1)
    check("left child is 0, right child is 2", left[1] == 0 and right[1] == 2)

    # ---- 8. sorted-increasing sequence -> right-leaning path ----------------------------
    a = [1, 2, 3, 4, 5]
    root, left, right, parent = build_cartesian_tree(a)
    check("increasing -> root 0 with right spine", root == 0 and right[0] == 1 and right[1] == 2)

    # ---- 9. sorted-decreasing sequence -> left-leaning path -----------------------------
    a = [5, 4, 3, 2, 1]
    root, left, right, parent = build_cartesian_tree(a)
    check("decreasing -> root last with left spine", root == 4 and left[4] == 3 and left[3] == 2)

    # ---- 10. single element and duplicate-free two-element -------------------------------
    root, left, right, parent = build_cartesian_tree([7])
    check("single element", root == 0 and left == [-1] and right == [-1])
    r2 = CartesianRMQ([9, 4])
    check("two elements min", r2.min_value(0, 1) == 4 and r2.min_index(0, 1) == 1)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
