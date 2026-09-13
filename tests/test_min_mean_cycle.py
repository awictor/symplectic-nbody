"""Tests for Karp min mean cycle: matches brute force, negative-cycle certificate, recovered cycle."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from min_mean_cycle import (  # noqa: E402
    min_mean_cycle,
    cycle_mean,
    brute_min_mean_cycle,
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


def _bellman_has_negative_cycle(n, edges):
    """Independent negative-cycle detector."""
    dist = [0.0] * n  # start all at 0 = virtual source to all
    for _ in range(n):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v] - 1e-12:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            return False
    # one more pass: any relaxation => negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v] - 1e-12:
            return True
    return False


def _is_real_cycle(cycle, edges):
    eset = {(u, v) for u, v, _ in edges}
    m = len(cycle)
    if m == 0:
        return False
    if len(set(cycle)) != m:
        return False  # must be simple (no repeated vertex)
    for i in range(m):
        if (cycle[i], cycle[(i + 1) % m]) not in eset:
            return False
    return True


def main():
    # ---- 1. matches brute force on random small graphs ----------------------------------
    rng = _lcg(2024)
    mismatches = 0
    tested = 0
    acyclic_agree = 0
    for _ in range(300):
        n = 2 + rng() % 5
        edges = []
        for u in range(n):
            for v in range(n):
                if u != v and rng() % 100 < 40:
                    w = (rng() % 21) - 10  # weights in [-10, 10]
                    edges.append((u, v, float(w)))
        mean, cyc = min_mean_cycle(n, edges)
        bmean, bcyc = brute_min_mean_cycle(n, edges)
        tested += 1
        if mean is None and bmean is None:
            acyclic_agree += 1
            continue
        if mean is None or bmean is None:
            mismatches += 1
        elif abs(mean - bmean) > 1e-9:
            mismatches += 1
    check("min mean == brute force (300 random graphs)", mismatches == 0,
          f"{mismatches}/{tested} mismatched")
    check("acyclic graphs agreed (both None)", acyclic_agree > 0, f"{acyclic_agree} seen")

    # ---- 2. recovered cycle is real and has the reported mean ---------------------------
    rng = _lcg(77)
    bad_cycle = bad_mean = 0
    checked = 0
    for _ in range(300):
        n = 3 + rng() % 5
        edges = []
        for u in range(n):
            for v in range(n):
                if u != v and rng() % 100 < 45:
                    w = (rng() % 21) - 10
                    edges.append((u, v, float(w)))
        mean, cyc = min_mean_cycle(n, edges)
        if mean is None:
            continue
        checked += 1
        if not _is_real_cycle(cyc, edges):
            bad_cycle += 1
        elif abs(cycle_mean(cyc, edges) - mean) > 1e-9:
            bad_mean += 1
    check("recovered cycle is a real simple cycle", bad_cycle == 0, f"{bad_cycle} bad")
    check("recovered cycle mean equals reported lambda*", bad_mean == 0, f"{bad_mean} bad")
    check("actually recovered cycles in the corpus", checked > 50, f"{checked}")

    # ---- 3. negative-cycle certificate matches Bellman-Ford -----------------------------
    rng = _lcg(555)
    mism = 0
    neg_seen = 0
    for _ in range(300):
        n = 2 + rng() % 5
        edges = []
        for u in range(n):
            for v in range(n):
                if u != v and rng() % 100 < 40:
                    w = (rng() % 21) - 10
                    edges.append((u, v, float(w)))
        mean, _ = min_mean_cycle(n, edges)
        has_neg = _bellman_has_negative_cycle(n, edges)
        karp_neg = mean is not None and mean < -1e-9
        if has_neg != karp_neg:
            mism += 1
        if has_neg:
            neg_seen += 1
    check("min mean < 0 iff negative cycle exists (Bellman-Ford)", mism == 0, f"{mism} mismatched")
    check("negative cycles present in corpus", neg_seen > 0, f"{neg_seen}")

    # ---- 4. hand examples ---------------------------------------------------------------
    # a 3-cycle 0->1->2->0 with weights 1,2,3 -> mean 2
    edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)]
    mean, cyc = min_mean_cycle(3, edges)
    check("triangle mean = 2.0", abs(mean - 2.0) < 1e-9, f"{mean}")
    check("triangle cycle length 3", len(cyc) == 3)

    # two cycles: a cheap 2-cycle (mean 0.5) and an expensive triangle
    edges = [(0, 1, 0.0), (1, 0, 1.0),          # 2-cycle mean 0.5
             (1, 2, 5.0), (2, 3, 5.0), (3, 1, 5.0)]  # triangle mean 5
    mean, cyc = min_mean_cycle(4, edges)
    check("picks the cheaper 2-cycle (mean 0.5)", abs(mean - 0.5) < 1e-9, f"{mean}")

    # a self-loop is the tightest cycle
    edges = [(0, 0, -3.0), (0, 1, 1.0), (1, 0, 1.0)]
    mean, cyc = min_mean_cycle(2, edges)
    check("self-loop mean = -3.0", abs(mean - (-3.0)) < 1e-9, f"{mean}")

    # a DAG has no cycle
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    mean, cyc = min_mean_cycle(3, edges)
    check("DAG -> no cycle (None)", mean is None)

    # empty graph
    check("no edges -> None", min_mean_cycle(3, [])[0] is None)
    check("n=0 -> None", min_mean_cycle(0, [])[0] is None)

    # out of bounds
    try:
        min_mean_cycle(2, [(0, 3, 1.0)])
        check("out-of-bounds raises", False)
    except ValueError:
        check("out-of-bounds raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
