"""Tests for hits: hub/authority roles, normalization, eigenvector match, structure."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hits import (hits, top_hubs, top_authorities, authority_matrix, hub_matrix,
                  adjacency_matrix)
from eigen import power_iteration

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


def _norm(d):
    return math.sqrt(sum(v * v for v in d.values()))


# --- hub-and-spoke: H links to A,B,C; X to A,B; Y to A ---------------------
g = {"H": ["A", "B", "C"], "X": ["A", "B"], "Y": ["A"]}
hub, auth = hits(g)

check("H is the top hub (links to the most authorities)", top_hubs(hub, 1)[0][0] == "H")
check("A is the top authority (most in-links)", top_authorities(auth, 1)[0][0] == "A")
# authority order follows in-degree here: A(3) > B(2) > C(1)
check("authority order A > B > C", auth["A"] > auth["B"] > auth["C"])
check("hub order H > X > Y", hub["H"] > hub["X"] > hub["Y"])

# --- scores are unit-normalized --------------------------------------------
check("hub scores unit norm", approx(_norm(hub), 1.0, 1e-6))
check("authority scores unit norm", approx(_norm(auth), 1.0, 1e-6))

# --- pure authorities have zero hub score, pure hubs zero authority --------
check("A (no out-links) has zero hub score", approx(hub["A"], 0.0, 1e-9))
check("H (no in-links) has zero authority score", approx(auth["H"], 0.0, 1e-9))

# --- all nodes (including target-only) appear ------------------------------
check("all nodes scored", set(hub) == {"A", "B", "C", "H", "X", "Y"})
check("scores are nonnegative", all(v >= -1e-12 for v in hub.values())
      and all(v >= -1e-12 for v in auth.values()))

# --- authority vector matches the dominant eigenvector of A'A --------------
M_auth, nodes = authority_matrix(g)
lam_a, va, _ = power_iteration(M_auth)
ev_auth = {nodes[i]: abs(va[i]) for i in range(len(nodes))}
na = _norm(ev_auth)
ev_auth = {k: v / na for k, v in ev_auth.items()}
check("authority = dominant eigenvector of A'A",
      all(approx(auth[k], ev_auth[k], 1e-3) for k in nodes))

# --- hub vector matches the dominant eigenvector of AA' --------------------
M_hub, nodes2 = hub_matrix(g)
lam_h, vh, _ = power_iteration(M_hub)
ev_hub = {nodes2[i]: abs(vh[i]) for i in range(len(nodes2))}
nh = _norm(ev_hub)
ev_hub = {k: v / nh for k, v in ev_hub.items()}
check("hub = dominant eigenvector of AA'",
      all(approx(hub[k], ev_hub[k], 1e-3) for k in nodes2))

# --- adjacency matrix is correct -------------------------------------------
A, an = adjacency_matrix(g)
idx = {n: i for i, n in enumerate(an)}
check("adjacency has the H->A edge", A[idx["H"]][idx["A"]] == 1.0)
check("adjacency lacks a non-edge", A[idx["A"]][idx["H"]] == 0.0)
check("adjacency row sum = out-degree", sum(A[idx["H"]]) == 3)

# --- a bipartite hub/authority structure: hubs link to a shared authority set
bip = {"h1": ["a1", "a2", "a3"], "h2": ["a1", "a2", "a3"], "h3": ["a1", "a2"]}
hb, ab = hits(bip)
# h1 and h2 link to the same 3 -> equal, strongest hubs
check("identical hubs get equal scores", approx(hb["h1"], hb["h2"], 1e-6))
check("h1/h2 outrank h3", hb["h1"] > hb["h3"])
# a1,a2 linked by all three hubs -> outrank a3 (linked by two)
check("more-linked authority ranks higher", ab["a1"] > ab["a3"])
check("a1 and a2 tie", approx(ab["a1"], ab["a2"], 1e-6))

# --- a directed cycle is symmetric -----------------------------------------
cyc = {0: [1], 1: [2], 2: [0]}
hc, ac = hits(cyc)
check("cycle hubs all equal", approx(hc[0], hc[1], 1e-6) and approx(hc[1], hc[2], 1e-6))
check("cycle authorities all equal", approx(ac[0], ac[1], 1e-6))

# --- empty graph -----------------------------------------------------------
check("empty graph -> empty scores", hits({}) == ({}, {}))

# --- single edge -----------------------------------------------------------
he, ae = hits({"u": ["v"]})
check("single edge: u is the hub", he["u"] > he["v"])
check("single edge: v is the authority", ae["v"] > ae["u"])

# --- determinism -----------------------------------------------------------
check("HITS deterministic", hits(g) == hits(g))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all hits tests passed")
