"""Tests for heavy-light decomposition: path queries match naive walk; LCA matches; chains O(log n)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heavy_light import HeavyLight, naive_path  # noqa: E402


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


def _random_tree(n, rng):
    parent = [-1] * n
    edges = []
    for i in range(1, n):
        p = rng() % i
        parent[i] = p
        edges.append((p, i))
    return parent, edges


def _naive_lca(u, v, parent):
    au = set()
    x = u
    while x != -1:
        au.add(x)
        x = parent[x]
    x = v
    while x != -1:
        if x in au:
            return x
        x = parent[x]
    return -1


def main():
    # ---- 1. path sum / max match naive walk over many trees, all pairs ------------------
    rng = _lcg(2024)
    mism_sum = mism_max = mism_lca = 0
    for _ in range(80):
        n = 2 + rng() % 12
        parent, edges = _random_tree(n, rng)
        vals = [(rng() % 41) - 20 for _ in range(n)]
        hld = HeavyLight(n, edges, values=vals, root=0)
        depth = hld.depth
        for u in range(n):
            for v in range(n):
                nodes = naive_path(u, v, parent, depth)
                exp_sum = sum(vals[x] for x in nodes)
                exp_max = max(vals[x] for x in nodes)
                if hld.path_sum(u, v) != exp_sum:
                    mism_sum += 1
                if hld.path_max(u, v) != exp_max:
                    mism_max += 1
                if hld.lca(u, v) != _naive_lca(u, v, parent):
                    mism_lca += 1
    check("path_sum matches naive walk (80 trees, all pairs)", mism_sum == 0, f"{mism_sum}")
    check("path_max matches naive walk", mism_max == 0, f"{mism_max}")
    check("HLD lca matches naive", mism_lca == 0, f"{mism_lca}")

    # ---- 2. updates then queries --------------------------------------------------------
    rng = _lcg(77)
    mism = 0
    for _ in range(60):
        n = 2 + rng() % 12
        parent, edges = _random_tree(n, rng)
        vals = [(rng() % 21) - 10 for _ in range(n)]
        hld = HeavyLight(n, edges, values=vals, root=0)
        depth = hld.depth
        # apply some updates, keeping a shadow copy
        for _ in range(10):
            v = rng() % n
            if rng() % 2:
                nv = (rng() % 41) - 20
                vals[v] = nv
                hld.update(v, nv)
            else:
                d = (rng() % 11) - 5
                vals[v] += d
                hld.add(v, d)
        for _ in range(15):
            u, w = rng() % n, rng() % n
            nodes = naive_path(u, w, parent, depth)
            if hld.path_sum(u, w) != sum(vals[x] for x in nodes):
                mism += 1
            if hld.path_max(u, w) != max(vals[x] for x in nodes):
                mism += 1
    check("path queries correct after updates", mism == 0, f"{mism}")

    # ---- 3. chains are O(log n) ---------------------------------------------------------
    rng = _lcg(7)
    ok = True
    worst_ratio = 0.0
    for _ in range(40):
        n = 4 + rng() % 60
        parent, edges = _random_tree(n, rng)
        hld = HeavyLight(n, edges, root=0)
        light = hld.max_light_edges_on_any_root_path()
        bound = math.log2(n) + 1
        if light > bound + 1e-9:
            ok = False
        worst_ratio = max(worst_ratio, light / bound if bound > 0 else 0)
    check("max light edges on any root path <= log2(n)+1", ok, f"ratio {worst_ratio:.2f}")

    # ---- 4. positions form a valid permutation, each chain contiguous -------------------
    rng = _lcg(321)
    ok_perm = ok_contig = True
    for _ in range(40):
        n = 3 + rng() % 20
        parent, edges = _random_tree(n, rng)
        hld = HeavyLight(n, edges, root=0)
        if sorted(hld.pos) != list(range(n)):
            ok_perm = False
        # each chain's vertices occupy a contiguous position block
        chains = {}
        for v in range(n):
            chains.setdefault(hld.head[v], []).append(hld.pos[v])
        for h, positions in chains.items():
            positions.sort()
            if positions != list(range(positions[0], positions[0] + len(positions))):
                ok_contig = False
    check("positions form a permutation of 0..n-1", ok_perm)
    check("each chain occupies contiguous positions", ok_contig)

    # ---- 5. hand example: a path graph 0-1-2-3-4 ----------------------------------------
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    hld = HeavyLight(5, edges, values=[1, 2, 3, 4, 5], root=0)
    check("path graph sum(0..4) = 15", hld.path_sum(0, 4) == 15)
    check("path graph max(0..4) = 5", hld.path_max(0, 4) == 5)
    check("path graph sum(1..3) = 9", hld.path_sum(1, 3) == 9)
    hld.update(2, 100)
    check("after update, max(0..4) = 100", hld.path_max(0, 4) == 100)
    check("after update, sum(0..4) = 112", hld.path_sum(0, 4) == 112)

    # ---- 6. star graph ------------------------------------------------------------------
    # center 0 connected to 1,2,3,4
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    hld = HeavyLight(5, edges, values=[10, 1, 2, 3, 4], root=0)
    check("star: sum(1,3) = 1+10+3 = 14", hld.path_sum(1, 3) == 14)
    check("star: max(2,4) = 10 (center)", hld.path_max(2, 4) == 10)
    check("star: single vertex sum(3,3)=3", hld.path_sum(3, 3) == 3)

    # ---- 7. edge cases ------------------------------------------------------------------
    single = HeavyLight(1, [], values=[42], root=0)
    check("single node sum = 42", single.path_sum(0, 0) == 42)
    check("single node lca(0,0)=0", single.lca(0, 0) == 0)
    try:
        HeavyLight(0, [])
        check("n=0 raises", False)
    except ValueError:
        check("n=0 raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
