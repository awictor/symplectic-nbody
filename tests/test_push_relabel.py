"""Tests for push_relabel: agrees with Dinic, flow conservation, max-flow min-cut, bipartite."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from push_relabel import PushRelabel, max_flow, bipartite_matching  # noqa: E402
import dinic  # noqa: E402


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

    def randint(self, n):
        return (self.nxt() >> 8) % n


def main():
    rng = LCG(2024)

    # ---- 1. hand-built network with known optimum -------------------------------------
    # classic CLRS network, max flow = 23
    edges = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12),
             (3, 2, 9), (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    check("CLRS network max flow == 23", max_flow(6, edges, 0, 5) == 23, f"{max_flow(6, edges, 0, 5)}")

    # simple series/parallel
    check("single edge bottleneck", max_flow(3, [(0, 1, 5), (1, 2, 3)], 0, 2) == 3)
    check("parallel paths sum", max_flow(4, [(0, 1, 4), (1, 3, 4), (0, 2, 6), (2, 3, 6)], 0, 3) == 10)

    # ---- 2. agreement with the independent Dinic solver over random networks ----------
    mism = 0
    trials = 300
    for _ in range(trials):
        n = 4 + rng.randint(6)      # 4..9 nodes
        m = rng.randint(n * 2) + n
        elist = []
        for _ in range(m):
            u = rng.randint(n)
            v = rng.randint(n)
            if u != v:
                elist.append((u, v, 1 + rng.randint(12)))
        s = 0
        t = n - 1
        pr_val = max_flow(n, elist, s, t)
        dn_val = dinic.max_flow(n, elist, s, t)
        if pr_val != dn_val:
            mism += 1
            if mism <= 5:
                print(f"    MISMATCH n={n} pr={pr_val} dinic={dn_val} edges={elist}")
    check(f"push-relabel agrees with Dinic over {trials} random networks", mism == 0,
          f"{mism} mismatches")

    # ---- 3. flow conservation and capacity limits -------------------------------------
    pr = PushRelabel(6)
    eidx = [pr.add_edge(u, v, c) for u, v, c in edges]
    val = pr.max_flow(0, 5)
    # conservation at internal nodes
    inflow = [0] * 6
    outflow = [0] * 6
    for i, (u, v, c) in enumerate(edges):
        f = pr.flow_on(eidx[i])
        check_cap = 0 <= f <= c
        if not check_cap:
            break
        outflow[u] += f
        inflow[v] += f
    cap_ok = all(0 <= pr.flow_on(eidx[i]) <= edges[i][2] for i in range(len(edges)))
    check("no edge exceeds capacity", cap_ok)
    conserve_ok = all(inflow[v] == outflow[v] for v in range(6) if v not in (0, 5))
    check("flow conserved at internal nodes", conserve_ok,
          f"in={inflow} out={outflow}")
    check("net outflow from source == flow value", outflow[0] - inflow[0] == val)

    # ---- 4. max-flow min-cut theorem --------------------------------------------------
    for _ in range(50):
        n = 5 + rng.randint(5)
        elist = []
        for _ in range(n * 2):
            u, v = rng.randint(n), rng.randint(n)
            if u != v:
                elist.append((u, v, 1 + rng.randint(10)))
        pr = PushRelabel(n)
        for u, v, c in elist:
            pr.add_edge(u, v, c)
        flow = pr.max_flow(0, n - 1)
        cut = set(pr.min_cut(0))
        # capacity of edges crossing the cut (source side -> sink side)
        cut_cap = sum(c for u, v, c in elist if u in cut and v not in cut)
        if not (flow == cut_cap and 0 in cut and (n - 1) not in cut):
            check("max-flow min-cut", False, f"flow={flow} cut_cap={cut_cap}")
            break
    else:
        check("max-flow equals min-cut capacity, cut separates s and t", True)

    # ---- 5. bipartite matching vs brute force -----------------------------------------
    def brute_matching(nl, nr, pairs):
        adj = [[] for _ in range(nl)]
        for li, rj in pairs:
            adj[li].append(rj)
        match_r = [-1] * nr

        def try_aug(u, seen):
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    if match_r[v] == -1 or try_aug(match_r[v], seen):
                        match_r[v] = u
                        return True
            return False

        count = 0
        for u in range(nl):
            if try_aug(u, [False] * nr):
                count += 1
        return count

    mism_b = 0
    for _ in range(60):
        nl = 2 + rng.randint(4)
        nr = 2 + rng.randint(4)
        pairs = set()
        for _ in range(rng.randint(nl * nr)):
            pairs.add((rng.randint(nl), rng.randint(nr)))
        pairs = list(pairs)
        if bipartite_matching(nl, nr, pairs) != brute_matching(nl, nr, pairs):
            mism_b += 1
    check("bipartite matching matches brute force", mism_b == 0, f"{mism_b} mismatches")

    # ---- 6. edge cases ---------------------------------------------------------------
    check("source == sink -> 0", max_flow(3, [(0, 1, 5)], 0, 0) == 0)
    check("no path -> 0 flow", max_flow(4, [(0, 1, 5), (2, 3, 5)], 0, 3) == 0)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
