"""Tests for lru_cache: eviction order, O(1) correctness vs brute force, LFU, hit rates."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lru_cache import LRUCache, LFUCache

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- LRU basic behaviour ---------------------------------------------------
c = LRUCache(2)
c.put(1, "a")
c.put(2, "b")
check("get returns stored value", c.get(1) == "a")
c.put(3, "c")                      # 1 was just used, so 2 is LRU and evicted
check("evicts least-recently-used", 2 not in c)
check("keeps recently-used", 1 in c and 3 in c)
check("MRU-to-LRU order after access", c.keys_mru_to_lru() == [3, 1])

# --- miss returns default, counts a miss -----------------------------------
c2 = LRUCache(2)
check("miss returns default", c2.get(99, "x") == "x")
check("miss counted", c2.misses == 1 and c2.hits == 0)

# --- updating a key refreshes it and its value -----------------------------
c3 = LRUCache(2)
c3.put(1, "a")
c3.put(2, "b")
c3.put(1, "A")                     # update 1 -> now MRU
c3.put(3, "c")                     # evicts 2
check("update changes value", c3.get(1) == "A")
check("update also promotes recency", 2 not in c3 and 3 in c3)

# --- capacity is never exceeded --------------------------------------------
c4 = LRUCache(3)
for i in range(20):
    c4.put(i, i)
check("capacity never exceeded", len(c4) == 3)
check("keeps the last three inserted", set(c4.keys_mru_to_lru()) == {17, 18, 19})

# --- invalid capacity ------------------------------------------------------
def raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


check("zero capacity raises", raises(lambda: LRUCache(0)))

# --- LRU matches a brute-force reference over random workloads --------------
def brute_lru_keys(cap, ops):
    order = []
    store = set()
    for op, k in ops:
        if op == "g":
            if k in store:
                order.remove(k)
                order.append(k)
        else:  # put
            if k in store:
                order.remove(k)
            elif len(store) >= cap:
                ev = order.pop(0)
                store.discard(ev)
            store.add(k)
            order.append(k)
    return store


state = 5


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


all_match = True
for trial in range(60):
    ops = []
    for _ in range(120):
        ops.append(("g" if rng() < 0.4 else "p", int(rng() * 8)))
    cap = 3
    lru = LRUCache(cap)
    for op, k in ops:
        if op == "g":
            lru.get(k)
        else:
            lru.put(k, k)
    if set(lru.map) != brute_lru_keys(cap, ops):
        all_match = False
check("LRU matches brute force over 60 workloads", all_match)

# --- hit rate accounting ---------------------------------------------------
c5 = LRUCache(2)
c5.put(1, 1)
c5.get(1)          # hit
c5.get(2)          # miss
c5.get(1)          # hit
check("hit rate computed", abs(c5.hit_rate() - 2 / 3) < 1e-9)

# --- LFU basic behaviour ---------------------------------------------------
f = LFUCache(2)
f.put(1, "a")
f.put(2, "b")
f.get(1)
f.get(1)           # 1 freq=3, 2 freq=1
f.put(3, "c")      # evicts 2 (least frequent)
check("LFU evicts least-frequent", 2 not in f)
check("LFU keeps frequent", 1 in f and 3 in f)
check("LFU tracks frequency", f.frequency(1) == 3)

# --- LFU breaks frequency ties by recency (oldest goes) --------------------
f2 = LFUCache(2)
f2.put(1, "a")     # freq 1
f2.put(2, "b")     # freq 1, both tied; 1 is older
f2.put(3, "c")     # evict the older of the freq-1 items -> 1
check("LFU tie-break evicts oldest", 1 not in f2 and 2 in f2 and 3 in f2)

# --- LFU capacity respected ------------------------------------------------
f3 = LFUCache(3)
for i in range(20):
    f3.put(i, i)
check("LFU capacity never exceeded", len(f3) == 3)

# --- LFU wins on a skewed (hot-key) workload -------------------------------
def zipf(n):
    r = rng()
    return int(n * (r ** 3))       # biased toward small indices


cap = 10
N = 200
lru_s = LRUCache(cap)
lfu_s = LFUCache(cap)
for _ in range(5000):
    k = zipf(N)
    if lru_s.get(k) is None:
        lru_s.put(k, k)
    if lfu_s.get(k) is None:
        lfu_s.put(k, k)
check("LFU beats LRU on a skewed workload", lfu_s.hit_rate() > lru_s.hit_rate())
check("both caches achieve some hits", lru_s.hit_rate() > 0 and lfu_s.hit_rate() > 0)

# --- a repeated single key is always a hit after the first -----------------
c6 = LRUCache(4)
c6.put(7, 7)
hits_before = c6.hits
for _ in range(100):
    c6.get(7)
check("repeated key always hits", c6.hits - hits_before == 100)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all lru_cache tests passed")
