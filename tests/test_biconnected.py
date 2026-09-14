"""Tests for biconnected components: edge partition, articulation cross-checks, known decompositions."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import biconnected as B  # noqa: E402


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
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _art_from_blocks(blocks):
    from collections import Counter
    vc = Counter()
    for blk in blocks:
        verts = set()
        for (u, v) in blk:
            verts.add(u)
            verts.add(v)
        for x in verts:
            vc[x] += 1
    return set(x for x, c in vc.items() if c >= 2)


def main():
    # ---- 1. a single cycle is one block with no articulation points ---------------------
    blocks, art = B.biconnected_components(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    check("cycle: 1 block", len(blocks) == 1)
    check("cycle: no articulation points", art == set())

    # ---- 2. a path (tree) has every edge as its own block, internal verts as cut points -
    blocks, art = B.biconnected_components(4, [(0, 1), (1, 2), (2, 3)])
    check("path: 3 blocks", len(blocks) == 3)
    check("path: internal vertices are cut points", art == {1, 2})

    # ---- 3. a star: center is a cut vertex, each spoke its own block --------------------
    blocks, art = B.biconnected_components(4, [(0, 1), (0, 2), (0, 3)])
    check("star: 3 blocks", len(blocks) == 3)
    check("star: center is the only articulation point", art == {0})

    # ---- 4. two triangles sharing a vertex (bowtie) ------------------------------------
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]
    blocks, art = B.biconnected_components(5, edges)
    check("bowtie: 2 blocks", len(blocks) == 2)
    check("bowtie: shared vertex is the articulation point", art == {2})

    # ---- 5. the blocks partition the edges exactly --------------------------------------
    for n, es in [(5, edges), (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
                  (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])]:
        blocks, _ = B.biconnected_components(n, es)
        total = sum(len(b) for b in blocks)
        seen = set()
        dup = False
        for b in blocks:
            for e in b:
                fe = frozenset(e)
                if fe in seen:
                    dup = True
                seen.add(fe)
        check(f"blocks partition {len(es)} edges (no dup, count matches)",
              total == len(es) and not dup and len(seen) == len(es), f"total {total}")

    # ---- 6. articulation == "in two or more blocks" ------------------------------------
    blocks, art = B.biconnected_components(5, edges)
    check("articulation points are exactly the multi-block vertices", _art_from_blocks(blocks) == art)

    # ---- 7. articulation matches brute-force deletion test ------------------------------
    for n, es in [(5, edges), (4, [(0, 1), (1, 2), (2, 3)]), (4, [(0, 1), (0, 2), (0, 3)])]:
        _, art = B.biconnected_components(n, es)
        check(f"articulation matches brute force (n={n})", art == B._brute_articulation(n, es))

    # ---- 8. randomized cross-check ------------------------------------------------------
    rnd = _lcg(7)
    mism = 0
    for _ in range(80):
        n = 6 + int(rnd() * 6)
        eset = set()
        for _ in range(int(n * 1.5)):
            u = int(rnd() * n)
            v = int(rnd() * n)
            if u != v:
                eset.add(frozenset((u, v)))
        es = [tuple(e) for e in eset]
        blocks, art = B.biconnected_components(n, es)
        if art != B._brute_articulation(n, es):
            mism += 1
        elif _art_from_blocks(blocks) != art:
            mism += 1
        elif sum(len(b) for b in blocks) != len(es):
            mism += 1
    check("randomized: articulation + partition all consistent", mism == 0, f"{mism}")

    # ---- 9. block-cut tree is acyclic with the right node count -------------------------
    bnodes, cnodes, tree_edges = B.block_cut_tree(5, edges)
    # block-cut tree: |V| = #blocks + #cut vertices, |E| = |V| - (#connected components of tree)
    nverts = len(bnodes) + len(cnodes)
    check("block-cut tree node count", nverts == len(blocks) + len(art))
    # for a connected graph the block-cut tree is a tree: edges == nodes - 1
    check("block-cut tree is a tree (edges == nodes - 1)", len(tree_edges) == nverts - 1,
          f"E {len(tree_edges)} V {nverts}")

    # ---- 10. K4 (complete graph on 4) is one biconnected block --------------------------
    k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    blocks, art = B.biconnected_components(4, k4)
    check("K4 is a single block with no articulation points", len(blocks) == 1 and art == set())

    # ---- 11. two disconnected cycles: two blocks, no articulation ----------------------
    two = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    blocks, art = B.biconnected_components(6, two)
    check("two disjoint cycles: 2 blocks, no articulation", len(blocks) == 2 and art == set())

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
