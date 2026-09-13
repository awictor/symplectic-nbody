"""Tests for dominator tree: tree dominance matches the deletion-based definition."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dominator_tree import DominatorTree, brute_dominates  # noqa: E402


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
        return state >> 8

    return nxt


def main():
    # ---- 1. tree dominance matches deletion definition on random CFGs -------------------
    rng = _lcg(2024)
    mismatches = 0
    pairs = 0
    for _ in range(200):
        n = 2 + rng() % 7
        entry = 0
        edges = []
        # bias toward reachability: chain 0->1->...->n-1 plus random extra edges
        for i in range(n - 1):
            if rng() % 100 < 70:
                edges.append((i, i + 1))
        for _ in range(n):
            u = rng() % n
            v = rng() % n
            if u != v:
                edges.append((u, v))
        dt = DominatorTree(n, edges, entry)
        for target in range(n):
            for d in range(n):
                bru = brute_dominates(n, edges, entry, d, target)
                if bru is None:
                    continue  # target unreachable: skip
                tree_says = dt.dominates(d, target)
                pairs += 1
                if tree_says != bru:
                    mismatches += 1
    check("tree dominance == deletion definition (200 CFGs)", mismatches == 0,
          f"{mismatches}/{pairs} mismatched")
    check("exercised many pairs", pairs > 500, f"{pairs}")

    # ---- 2. entry dominates all reachable; every node dominates itself ------------------
    rng = _lcg(99)
    ok_entry = ok_self = True
    for _ in range(60):
        n = 2 + rng() % 6
        edges = [(i, i + 1) for i in range(n - 1)]
        for _ in range(n):
            u, v = rng() % n, rng() % n
            if u != v:
                edges.append((u, v))
        dt = DominatorTree(n, edges, 0)
        idom = dt.immediate_dominators()
        for node in idom:
            if not dt.dominates(0, node):
                ok_entry = False
            if not dt.dominates(node, node):
                ok_self = False
    check("entry dominates every reachable node", ok_entry)
    check("every node dominates itself", ok_self)

    # ---- 3. tree has exactly (reachable - 1) edges --------------------------------------
    rng = _lcg(7)
    ok = True
    for _ in range(60):
        n = 2 + rng() % 6
        edges = [(i, i + 1) for i in range(n - 1)]  # fully reachable chain
        dt = DominatorTree(n, edges, 0)
        reach = sum(1 for r in dt.reachable if r)
        if len(dt.tree_edges()) != reach - 1:
            ok = False
    check("tree edges == reachable count - 1", ok)

    # ---- 4. hand example: diamond CFG ---------------------------------------------------
    #   0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3   (a classic if/else diamond)
    # idom(1)=0, idom(2)=0, idom(3)=0 (both branches reach 3, so neither dominates it)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    dt = DominatorTree(4, edges, 0)
    idom = dt.immediate_dominators()
    check("diamond: idom(1)=0", idom[1] == 0)
    check("diamond: idom(3)=0 (join not dominated by a branch)", idom[3] == 0)
    check("diamond: 1 does NOT dominate 3", not dt.dominates(1, 3))
    check("diamond: 0 dominates 3", dt.dominates(0, 3))

    # ---- 5. hand example: a chain with a bypass -----------------------------------------
    #   0 -> 1 -> 2 -> 3, plus 0 -> 3 (bypass). Then 1 and 2 do NOT dominate 3.
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    dt = DominatorTree(4, edges, 0)
    check("bypass: 1 does not dominate 3", not dt.dominates(1, 3))
    check("bypass: 1 dominates 2", dt.dominates(1, 2))
    check("bypass: dominators_of(2) = {0,1,2}", dt.dominators_of(2) == {0, 1, 2})

    # ---- 6. hand example: a loop --------------------------------------------------------
    #   0 -> 1 -> 2 -> 1 (back edge), 2 -> 3. Header 1 dominates 2 and 3.
    edges = [(0, 1), (1, 2), (2, 1), (2, 3)]
    dt = DominatorTree(4, edges, 0)
    check("loop: 1 dominates 3", dt.dominates(1, 3))
    check("loop: idom(3)=2", dt.immediate_dominators()[3] == 2)

    # ---- 7. unreachable nodes excluded --------------------------------------------------
    #   0 -> 1, and an isolated 2 -> 3 not reachable from 0
    edges = [(0, 1), (2, 3)]
    dt = DominatorTree(4, edges, 0)
    idom = dt.immediate_dominators()
    check("unreachable node 2 absent from idom", 2 not in idom)
    check("unreachable node 3 absent from idom", 3 not in idom)
    check("dominates on unreachable is False", not dt.dominates(0, 3))

    # ---- 8. edge cases ------------------------------------------------------------------
    single = DominatorTree(1, [], 0)
    check("single node: idom(0)=0", single.immediate_dominators()[0] == 0)
    check("single node: no tree edges", single.tree_edges() == [])
    try:
        DominatorTree(0, [])
        check("n=0 raises", False)
    except ValueError:
        check("n=0 raises", True)
    try:
        DominatorTree(3, [], entry=5)
        check("bad entry raises", False)
    except ValueError:
        check("bad entry raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
