"""Tests for louvain: modularity correctness, planted-community recovery, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from louvain import detect, modularity

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def clique(nodes, w=1.0):
    e = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            e.append((nodes[i], nodes[j], w))
    return e


# --- two cliques joined by a bridge: recover the two communities -----------
edges = clique([0, 1, 2, 3]) + clique([4, 5, 6, 7]) + [(3, 4, 1.0)]
comm, q = detect(8, edges, seed=1)
check("two-clique graph: clique 1 is one community", len(set(comm[:4])) == 1)
check("two-clique graph: clique 2 is one community", len(set(comm[4:])) == 1)
check("the two cliques are different communities", comm[0] != comm[4])
check("modularity is substantial for clear communities", q > 0.3)

# --- returned modularity matches a direct computation ----------------------
check("reported modularity matches direct computation",
      abs(q - modularity(8, edges, comm)) < 1e-12)

# --- four planted communities ----------------------------------------------
edges4 = []
groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
for g in groups:
    edges4 += clique(g)
# sparse bridges between consecutive groups
edges4 += [(3, 4, 1.0), (7, 8, 1.0), (11, 12, 1.0)]
comm4, q4 = detect(16, edges4, seed=2)
# each planted group should be internally one community
each_one = all(len(set(comm4[g[0]:g[0] + 4])) == 1 for g in groups)
check("four planted communities each recovered as one community", each_one)
check("four distinct communities found", len(set(comm4)) == 4)
check("four-community modularity is high", q4 > 0.5)

# --- a single clique is one community --------------------------------------
single = clique([0, 1, 2, 3, 4])
cs, qs = detect(5, single, seed=1)
check("a single clique is one community", len(set(cs)) == 1)

# --- a graph with no community structure (near-complete) has low modularity
complete = clique(list(range(6)))
cc, qc = detect(6, complete, seed=1)
check("a complete graph has low modularity", qc < 0.1)

# --- modularity of the trivial partition (all in one community) is 0 -------
allone = [0] * 8
check("all-in-one-community modularity is ~0", abs(modularity(8, edges, allone)) < 1e-9)

# --- modularity of all-singletons is negative ------------------------------
singletons = list(range(8))
check("all-singletons modularity is negative", modularity(8, edges, singletons) < 0)

# --- the found partition beats both trivial partitions ---------------------
check("Louvain partition beats all-in-one and all-singletons",
      q > modularity(8, edges, allone) and q > modularity(8, edges, singletons))

# --- modularity is in the valid range --------------------------------------
check("modularity within [-0.5, 1]", -0.5 - 1e-9 <= q <= 1 + 1e-9)

# --- a ring of cliques (community structure with a cycle of bridges) -------
ring_edges = []
rings = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
for g in rings:
    ring_edges += clique(g)
ring_edges += [(2, 3, 1.0), (5, 6, 1.0), (8, 9, 1.0), (11, 0, 1.0)]
cr, qr = detect(12, ring_edges, seed=3)
check("ring-of-triangles: each triangle is a community",
      all(len(set(cr[g[0]:g[0] + 3])) == 1 for g in rings))

# --- weighted edges respected ----------------------------------------------
# heavy edges inside groups, light between
w_edges = clique([0, 1, 2], w=10.0) + clique([3, 4, 5], w=10.0) + [(2, 3, 0.1)]
cw, qw = detect(6, w_edges, seed=1)
check("weighted graph recovers the heavy-edge groups", cw[0] == cw[1] == cw[2] and cw[3] == cw[4] == cw[5]
      and cw[0] != cw[3])

# --- empty and single-node graphs ------------------------------------------
check("single node: one community", detect(1, [], seed=1)[0] == [0])
check("empty graph handled", detect(0, [], seed=1) == ([], 0.0))

# --- reproducibility -------------------------------------------------------
c1, _ = detect(16, edges4, seed=5)
c2, _ = detect(16, edges4, seed=5)
check("same seed gives identical communities", c1 == c2)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all louvain tests passed")
