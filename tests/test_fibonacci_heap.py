"""Tests for fibonacci_heap: sorted extraction, decrease-key, merge, degree bound, Dijkstra."""

import heapq
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fibonacci_heap import FibonacciHeap, dijkstra

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 17
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- extract-min returns keys in sorted order ------------------------------
vals = [int(rng() * 10000) for _ in range(500)]
h = FibonacciHeap()
for v in vals:
    h.insert(v)
check("length matches inserts", len(h) == len(vals))
out = []
while not h.is_empty():
    out.append(h.extract_min()[0])
check("extract-min yields sorted order (500 random keys)", out == sorted(vals))
check("heap is empty after draining", h.is_empty() and len(h) == 0)

# --- find-min tracks the minimum -------------------------------------------
h = FibonacciHeap()
h.insert(5); h.insert(3); h.insert(8)
check("find-min is the smallest", h.find_min()[0] == 3)
h.insert(1)
check("find-min updates on insert", h.find_min()[0] == 1)

# --- values travel with keys -----------------------------------------------
h = FibonacciHeap()
h.insert(3, "c"); h.insert(1, "a"); h.insert(2, "b")
check("extract-min returns the paired value", h.extract_min() == (1, "a"))
check("next extract-min value", h.extract_min() == (2, "b"))

# --- decrease-key ----------------------------------------------------------
h = FibonacciHeap()
nodes = [h.insert(i * 10, i) for i in range(20)]
# force some structure with an extract
h.insert(-100)
h.extract_min()          # removes -100, consolidates
h.decrease_key(nodes[15], -50)
check("decrease-key moves the node to the front", h.find_min()[0] == -50)
# drain and confirm still sorted
remaining = []
while not h.is_empty():
    remaining.append(h.extract_min()[0])
check("sorted after decrease-key", remaining == sorted(remaining))
# decrease-key must reject an increase
hz = FibonacciHeap()
nz = hz.insert(5)
raised = False
try:
    hz.decrease_key(nz, 10)
except ValueError:
    raised = True
check("decrease-key rejects an increase", raised)


# --- delete ----------------------------------------------------------------
h = FibonacciHeap()
handles = {v: h.insert(v) for v in [10, 20, 30, 40, 50]}
h.delete(handles[30])
out = []
while not h.is_empty():
    out.append(h.extract_min()[0])
check("delete removes exactly the target", out == [10, 20, 40, 50])

# --- merge -----------------------------------------------------------------
a = FibonacciHeap()
b = FibonacciHeap()
for v in [4, 8, 1, 9]:
    a.insert(v)
for v in [3, 7, 2, 6]:
    b.insert(v)
a.merge(b)
check("merge preserves count", len(a) == 8)
check("merged find-min", a.find_min()[0] == 1)
merged = []
while not a.is_empty():
    merged.append(a.extract_min()[0])
check("merged heap drains sorted", merged == [1, 2, 3, 4, 6, 7, 8, 9])

# --- against a binary heap over a random operation stream ------------------
fib = FibonacciHeap()
bh = []
ok = True
for _ in range(2000):
    op = rng()
    if op < 0.6 or not bh:
        k = int(rng() * 1000)
        fib.insert(k)
        heapq.heappush(bh, k)
    else:
        fk = fib.extract_min()[0]
        bk = heapq.heappop(bh)
        if fk != bk:
            ok = False
            break
check("matches a binary heap over a 2000-op random stream", ok)

# --- maximum root degree stays within the Fibonacci O(log n) bound ---------
h = FibonacciHeap()
for v in [int(rng() * 100000) for _ in range(1000)]:
    h.insert(v)
# trigger consolidation
h.extract_min()
n = len(h)
phi = (1 + math.sqrt(5)) / 2
bound = math.log(n) / math.log(phi) + 2
check(f"max root degree {h.max_root_degree()} within Fibonacci bound {bound:.1f}",
      h.max_root_degree() <= bound)

# --- Dijkstra on the Fibonacci heap matches a binary-heap Dijkstra ---------
def dijkstra_binary(n, edges, source):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [float("inf")] * n
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist


match = True
for _ in range(40):
    n = 8 + int(rng() * 8)
    edges = []
    for _ in range(n * 2):
        u = int(rng() * n)
        v = int(rng() * n)
        w = 1 + int(rng() * 20)
        if u != v:
            edges.append((u, v, w))
            edges.append((v, u, w))
    df = dijkstra(n, edges, 0)
    db = dijkstra_binary(n, edges, 0)
    if df != db:
        match = False
        break
check("Dijkstra (Fibonacci heap) matches Dijkstra (binary heap) over 40 random graphs", match)

# --- known small shortest-path -------------------------------------------
d = dijkstra(4, [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)], 0)
check("Dijkstra known distances", d == [0, 3, 1, 4])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all fibonacci_heap tests passed")
