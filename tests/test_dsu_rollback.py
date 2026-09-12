"""Tests for dsu_rollback: connectivity vs brute force, rollback correctness, nested snapshots."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dsu_rollback import RollbackDSU

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 555
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_connected(n, edges, a, b):
    """Fresh BFS/union over the given edges."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for u, v in edges:
        parent[find(u)] = find(v)
    return find(a) == find(b)


def brute_components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for u, v in edges:
        parent[find(u)] = find(v)
    return len({find(i) for i in range(n)})


# --- basic connectivity and component count --------------------------------
d = RollbackDSU(6)
check("initially 6 components", d.component_count() == 6)
d.unite(0, 1)
d.unite(2, 3)
check("after 2 unites: 4 components", d.component_count() == 4)
check("0 and 1 connected", d.connected(0, 1))
check("1 and 3 not connected", not d.connected(1, 3))

# --- unite returns whether it merged ---------------------------------------
check("uniting separate sets returns True", d.unite(1, 2))
check("uniting already-connected returns False", not d.unite(0, 3))  # 0-1-2-3 now one set

# --- rollback restores connectivity and count -----------------------------
d2 = RollbackDSU(8)
d2.unite(0, 1)
d2.unite(2, 3)
snap = d2.snapshot()
comps_before = d2.component_count()
d2.unite(1, 2)
d2.unite(4, 5)
d2.unite(6, 7)
check("state changed after unions", d2.component_count() != comps_before and d2.connected(0, 3))
d2.rollback(snap)
check("rollback restores component count", d2.component_count() == comps_before)
check("rollback undoes connectivity (0,3 separate again)", not d2.connected(0, 3))
check("rollback undoes later unions (4,5 separate)", not d2.connected(4, 5))
check("rollback preserves pre-snapshot unions (0,1 still connected)", d2.connected(0, 1))

# --- nested snapshots -------------------------------------------------------
d3 = RollbackDSU(10)
d3.unite(0, 1)
snap1 = d3.snapshot()
d3.unite(2, 3)
snap2 = d3.snapshot()
d3.unite(4, 5)
d3.unite(0, 2)
check("all unions applied", d3.connected(0, 3) and d3.connected(4, 5))
d3.rollback(snap2)
check("rollback to snap2: 4,5 undone but 2,3 kept", not d3.connected(4, 5) and d3.connected(2, 3))
check("rollback to snap2: 0,2 undone", not d3.connected(0, 2))
d3.rollback(snap1)
check("rollback to snap1: 2,3 undone", not d3.connected(2, 3))
check("rollback to snap1: 0,1 kept", d3.connected(0, 1))

# --- full rollback returns to all-singletons -------------------------------
d4 = RollbackDSU(20)
start = d4.snapshot()
for _ in range(30):
    d4.unite(int(rng() * 20), int(rng() * 20))
d4.rollback(start)
check("full rollback returns to 20 components", d4.component_count() == 20)
check("full rollback: no two elements connected",
      all(not d4.connected(i, j) for i in range(0, 20, 5) for j in range(i + 1, 20, 5)))

# --- connectivity always matches a brute-force recompute -------------------
ok = True
for _ in range(50):
    n = 5 + int(rng() * 10)
    dsu = RollbackDSU(n)
    live_edges = []
    for _ in range(n * 2):
        u, v = int(rng() * n), int(rng() * n)
        dsu.unite(u, v)
        live_edges.append((u, v))
    # check several pairs against brute force
    for _ in range(20):
        a, b = int(rng() * n), int(rng() * n)
        if dsu.connected(a, b) != brute_connected(n, live_edges, a, b):
            ok = False
            break
    if dsu.component_count() != brute_components(n, live_edges):
        ok = False
    if not ok:
        break
check("connectivity and component count match brute force over 50 graphs", ok)

# --- offline dynamic connectivity: add edges, query, retract ---------------
# a chain of snapshots simulating a segment-tree-over-time style workload
d5 = RollbackDSU(6)
snaps = []
edges_added = [(0, 1), (2, 3), (1, 2), (4, 5), (3, 4)]
for e in edges_added:
    snaps.append(d5.snapshot())
    d5.unite(*e)
# now everything except... 0-1-2-3-4-5 all connected via the chain
check("after all edges: 0 and 5 connected", d5.connected(0, 5))
check("one big component", d5.component_count() == 1)
# retract the last two edges
d5.rollback(snaps[3])   # undo (4,5) and (3,4)
check("after retracting 2 edges: 0 and 5 disconnected", not d5.connected(0, 5))
check("but 0 and 3 still connected", d5.connected(0, 3))

# --- redundant unions roll back cleanly ------------------------------------
d6 = RollbackDSU(4)
d6.unite(0, 1)
snap = d6.snapshot()
d6.unite(0, 1)   # redundant no-op
d6.unite(0, 1)   # redundant no-op
d6.rollback(snap)
check("redundant unions roll back without corrupting state",
      d6.connected(0, 1) and d6.component_count() == 3)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all dsu_rollback tests passed")
