"""Tests for Weisfeiler-Lehman: sound vs exact isomorphism, permutation-invariant, stable."""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weisfeiler_lehman import (  # noqa: E402
    refine,
    color_histogram,
    graph_hash,
    wl_kernel,
    possibly_isomorphic,
    is_isomorphic_bruteforce,
    relabel,
)


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


def _random_graph(n, rng, p=40):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng() % 100 < p:
                edges.append((u, v))
    return edges


def _random_perm(n, rng):
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng() % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def main():
    # ---- 1. soundness: WL never says "not isomorphic" for a truly isomorphic pair -------
    rng = _lcg(2024)
    false_negatives = 0
    iso_tested = 0
    for _ in range(300):
        n = 2 + rng() % 6
        e1 = _random_graph(n, rng)
        perm = _random_perm(n, rng)
        e2 = relabel(n, e1, perm)  # e2 is isomorphic to e1 by construction
        iso_tested += 1
        if not possibly_isomorphic(n, e1, n, e2):
            false_negatives += 1  # WL wrongly rejected an isomorphic pair
    check("WL never rejects an isomorphic pair (soundness, 300 pairs)",
          false_negatives == 0, f"{false_negatives} false negatives")
    check("exercised isomorphic pairs", iso_tested > 100)

    # ---- 2. isomorphic graphs share histogram and hash ----------------------------------
    rng = _lcg(77)
    ok_hist = ok_hash = True
    for _ in range(200):
        n = 3 + rng() % 6
        e1 = _random_graph(n, rng)
        perm = _random_perm(n, rng)
        e2 = relabel(n, e1, perm)
        if color_histogram(n, e1) != color_histogram(n, e2):
            ok_hist = False
        if graph_hash(n, e1) != graph_hash(n, e2):
            ok_hash = False
    check("isomorphic graphs share WL histogram", ok_hist)
    check("isomorphic graphs share WL hash", ok_hash)

    # ---- 3. WL agrees with exact on random (possibly non-iso) pairs ---------------------
    # When WL says "not isomorphic", exact must agree it's not. (WL false-positive only.)
    rng = _lcg(555)
    wl_wrong = 0
    wl_says_no = 0
    checked = 0
    for _ in range(400):
        n = 2 + rng() % 5
        e1 = _random_graph(n, rng)
        e2 = _random_graph(n, rng)
        wl = possibly_isomorphic(n, e1, n, e2)
        exact = is_isomorphic_bruteforce(n, e1, n, e2)
        checked += 1
        if not wl:
            wl_says_no += 1
            if exact:
                wl_wrong += 1  # WL rejected an actually-isomorphic pair: a real bug
    check("WL 'not isomorphic' is always correct (no false rejections)", wl_wrong == 0,
          f"{wl_wrong} wrong out of {wl_says_no} rejections")
    check("WL rejected many non-iso pairs (useful)", wl_says_no > 50, f"{wl_says_no}")

    # ---- 4. colouring is stable (one more round changes nothing) ------------------------
    rng = _lcg(7)
    ok_stable = True
    for _ in range(100):
        n = 2 + rng() % 8
        edges = _random_graph(n, rng)
        colors, rounds = refine(n, edges)
        # run one extra refinement round manually; the partition must not change
        colors2, _ = refine(n, edges, init=colors, max_rounds=1)
        # same partition -> same class structure
        part1 = {}
        for v, c in enumerate(colors):
            part1.setdefault(c, set()).add(v)
        part2 = {}
        for v, c in enumerate(colors2):
            part2.setdefault(c, set()).add(v)
        if sorted(map(sorted, part1.values())) != sorted(map(sorted, part2.values())):
            ok_stable = False
    check("stable colouring: one more round is idempotent", ok_stable)

    # ---- 5. permutation invariance of the histogram -------------------------------------
    rng = _lcg(321)
    ok = True
    for _ in range(100):
        n = 3 + rng() % 6
        edges = _random_graph(n, rng)
        h0 = color_histogram(n, edges)
        for _ in range(3):
            perm = _random_perm(n, rng)
            if color_histogram(n, relabel(n, edges, perm)) != h0:
                ok = False
    check("histogram invariant under relabelling", ok)

    # ---- 6. WL kernel: self-similarity is maximal, symmetric ----------------------------
    rng = _lcg(11)
    e = _random_graph(6, rng)
    e_other = _random_graph(6, rng)
    kaa = wl_kernel(6, e, 6, e)
    kab = wl_kernel(6, e, 6, e_other)
    kba = wl_kernel(6, e_other, 6, e)
    check("WL kernel symmetric", kab == kba)
    check("WL self-kernel >= cross-kernel", kaa >= kab)
    # isomorphic copy has identical self-kernel
    perm = _random_perm(6, rng)
    e_iso = relabel(6, e, perm)
    check("WL kernel equal for isomorphic copies", wl_kernel(6, e, 6, e) == wl_kernel(6, e_iso, 6, e_iso))

    # ---- 7. hand examples ---------------------------------------------------------------
    # two triangles (isomorphic)
    tri_a = [(0, 1), (1, 2), (2, 0)]
    tri_b = [(1, 2), (2, 0), (0, 1)]
    check("two triangles possibly-iso", possibly_isomorphic(3, tri_a, 3, tri_b))
    check("two triangles exactly iso", is_isomorphic_bruteforce(3, tri_a, 3, tri_b))

    # a path P3 vs a star (both 3 edges on 4 nodes but different) - actually P4 vs star K1,3
    path = [(0, 1), (1, 2), (2, 3)]          # degrees 1,2,2,1
    star = [(0, 1), (0, 2), (0, 3)]          # degrees 3,1,1,1
    check("path P4 vs star K1,3: WL says not iso", not possibly_isomorphic(4, path, 4, star))
    check("path P4 vs star K1,3: exact not iso", not is_isomorphic_bruteforce(4, path, 4, star))

    # the classic 1-WL fooling pair would need regular graphs; check WL is honest on C6 vs 2 triangles
    c6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]        # 6-cycle, all degree 2
    two_tri = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]   # two triangles, all degree 2
    # both 2-regular: 1-WL cannot tell them apart (histograms equal)
    check("C6 vs 2 triangles: WL cannot distinguish (both 2-regular)",
          possibly_isomorphic(6, c6, 6, two_tri))
    check("C6 vs 2 triangles: exact says NOT isomorphic (WL fooled honestly)",
          not is_isomorphic_bruteforce(6, c6, 6, two_tri))

    # different sizes
    check("different node counts -> not iso", not possibly_isomorphic(3, tri_a, 4, path))
    check("empty graphs iso", possibly_isomorphic(3, [], 3, []))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
