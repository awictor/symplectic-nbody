"""Tests for Hopcroft-Karp: matches Kuhn + brute force, Koenig duality, Hall witness."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hopcroft_karp import (  # noqa: E402
    HopcroftKarp,
    max_matching,
    maximum_matching_pairs,
    has_perfect_matching,
    hall_violator,
    kuhn_max_matching,
)

# cross-check against the max-flow bipartite matcher already in the repo
from dinic import bipartite_matching_size  # noqa: E402


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


def _brute_max_matching(nl, nr, edges):
    """Exhaustive: largest set of edges sharing no endpoint (small graphs only)."""
    edges = list(set(edges))
    best = 0
    # try all subsets by size, descending, stop at first valid of a given size
    for r in range(min(nl, nr), 0, -1):
        for combo in itertools.combinations(edges, r):
            ls = [u for u, _ in combo]
            rs = [v for _, v in combo]
            if len(set(ls)) == r and len(set(rs)) == r:
                return r
    return best


def _valid_matching(pairs):
    ls = [u for u, _ in pairs]
    rs = [v for _, v in pairs]
    return len(set(ls)) == len(ls) and len(set(rs)) == len(rs)


def main():
    # ---- 1. matches Kuhn, dinic-maxflow, and brute force on random graphs ---------------
    rng = _lcg(2024)
    mism_kuhn = mism_flow = mism_brute = 0
    trials = 0
    for _ in range(300):
        nl = 1 + rng() % 5
        nr = 1 + rng() % 5
        edges = []
        for u in range(nl):
            for v in range(nr):
                if rng() % 100 < 45:
                    edges.append((u, v))
        hk = max_matching(nl, nr, edges)
        if hk != kuhn_max_matching(nl, nr, edges):
            mism_kuhn += 1
        if hk != bipartite_matching_size(nl, nr, edges):
            mism_flow += 1
        if hk != _brute_max_matching(nl, nr, edges):
            mism_brute += 1
        trials += 1
    check("HK == Kuhn augmenting-path (300 graphs)", mism_kuhn == 0, f"{mism_kuhn} mismatched")
    check("HK == dinic max-flow matcher (300 graphs)", mism_flow == 0, f"{mism_flow} mismatched")
    check("HK == brute force (300 graphs)", mism_brute == 0, f"{mism_brute} mismatched")

    # ---- 2. returned pairs form a valid matching of the reported size -------------------
    rng = _lcg(99)
    ok_valid = ok_size = True
    for _ in range(100):
        nl = 2 + rng() % 5
        nr = 2 + rng() % 5
        edges = [(u, v) for u in range(nl) for v in range(nr) if rng() % 100 < 50]
        size = max_matching(nl, nr, edges)
        pairs = maximum_matching_pairs(nl, nr, edges)
        if not _valid_matching(pairs):
            ok_valid = False
        if len(pairs) != size:
            ok_size = False
        # every matched edge must actually exist
        eset = set(edges)
        if any((u, v) not in eset for u, v in pairs):
            ok_valid = False
    check("returned pairs form a valid matching", ok_valid)
    check("pair count equals reported matching size", ok_size)

    # ---- 3. Koenig: |max matching| == |min vertex cover|, and cover is genuine ----------
    rng = _lcg(7)
    ok_konig = ok_cover = True
    for _ in range(150):
        nl = 1 + rng() % 5
        nr = 1 + rng() % 5
        edges = list({(u, v) for u in range(nl) for v in range(nr) if rng() % 100 < 45})
        hk = HopcroftKarp(nl, nr, edges)
        m = hk.max_matching()
        lc, rc = hk.min_vertex_cover()
        if len(lc) + len(rc) != m:
            ok_konig = False
        # every edge must be covered by an endpoint in the cover
        for u, v in edges:
            if u not in lc and v not in rc:
                ok_cover = False
                break
    check("Koenig: |min vertex cover| == |max matching|", ok_konig)
    check("min vertex cover actually covers every edge", ok_cover)

    # ---- 4. perfect matching detection + Hall witness -----------------------------------
    # a left-perfect case: 3x3 identity-ish plus extras
    edges = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]
    check("perfect matching exists (3x3 with a system of distinct reps)",
          has_perfect_matching(3, 3, edges))
    check("no Hall violator when perfect", hall_violator(3, 3, edges) is None)

    # a deficient case: left {0,1,2} all point only at right {0,1} -> N(S)=2 < 3
    edges = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    check("no perfect matching when deficient", not has_perfect_matching(3, 2, edges))
    S = hall_violator(3, 2, edges)
    check("Hall violator found", S is not None and len(S) >= 1)
    if S is not None:
        neigh = set()
        eset = {}
        for u, v in edges:
            eset.setdefault(u, set()).add(v)
        for u in S:
            neigh |= eset.get(u, set())
        check("Hall violator has |N(S)| < |S|", len(neigh) < len(S), f"|S|={len(S)} |N|={len(neigh)}")

    # ---- 5. hand examples ---------------------------------------------------------------
    # complete bipartite K_{3,3}: matching 3
    edges = [(u, v) for u in range(3) for v in range(3)]
    check("K_{3,3} matching = 3", max_matching(3, 3, edges) == 3)
    # a path a0-b0-a1-b1-a2: matching 2 among 3 left / 2 right
    edges = [(0, 0), (1, 0), (1, 1), (2, 1)]
    check("path graph matching = 2", max_matching(3, 2, edges) == 2)
    # empty graph
    check("no edges -> matching 0", max_matching(3, 3, []) == 0)
    # single edge
    check("single edge -> matching 1", max_matching(2, 2, [(0, 1)]) == 1)
    # duplicate edges ignored
    check("duplicate edges deduped", max_matching(2, 2, [(0, 0), (0, 0), (1, 1)]) == 2)

    # ---- 6. out-of-bounds edge rejected -------------------------------------------------
    try:
        HopcroftKarp(2, 2, [(0, 5)])
        check("out-of-bounds edge raises", False)
    except ValueError:
        check("out-of-bounds edge raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
