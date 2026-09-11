"""Tests for skiplist: ordered map ops, brute-force agreement, range queries, level distribution."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skiplist import SkipList

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- basic insert / search / order -----------------------------------------
sl = SkipList(seed=1)
for k in [5, 2, 8, 1, 9, 3, 7]:
    sl.insert(k, k * 10)
check("keys iterate in sorted order", sl.keys() == [1, 2, 3, 5, 7, 8, 9])
check("search returns the stored value", sl.search(8) == 80)
check("search a missing key returns default", sl.search(4) is None)
check("search custom default", sl.search(4, -1) == -1)
check("contains present key", 3 in sl)
check("does not contain absent key", 4 not in sl)
check("length is correct", len(sl) == 7)
check("min and max", sl.min() == 1 and sl.max() == 9)

# --- items preserves values in order ---------------------------------------
check("items in sorted order with values",
      sl.items() == [(1, 10), (2, 20), (3, 30), (5, 50), (7, 70), (8, 80), (9, 90)])

# --- update does not duplicate ---------------------------------------------
added = sl.insert(5, 999)
check("re-insert updates value", sl.search(5) == 999)
check("re-insert returns False (no new key)", added is False)
check("re-insert does not change length", len(sl) == 7)

# --- deletion --------------------------------------------------------------
check("delete returns True for a present key", sl.delete(8))
check("deleted key is gone", 8 not in sl and sl.keys() == [1, 2, 3, 5, 7, 9])
check("length drops after delete", len(sl) == 6)
check("delete absent key returns False", not sl.delete(42))

# --- range queries ---------------------------------------------------------
check("range returns in-range pairs", sl.range(3, 7) == [(3, 30), (5, 999), (7, 70)])
check("range excludes out-of-range", all(3 <= k <= 7 for k, _ in sl.range(3, 7)))
check("range with no matches is empty", sl.range(100, 200) == [])
check("range covering all", len(sl.range(-10, 100)) == len(sl))

# --- empty list edge cases -------------------------------------------------
empty = SkipList(seed=2)
check("empty length 0", len(empty) == 0)
check("empty keys", empty.keys() == [])
check("empty search returns default", empty.search(1) is None)


def raises_keyerror(fn):
    try:
        fn()
        return False
    except KeyError:
        return True


check("min of empty raises", raises_keyerror(empty.min))
check("max of empty raises", raises_keyerror(empty.max))

# --- brute-force agreement over many random operations ---------------------
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


sl2 = SkipList(seed=7)
ref = {}
for _ in range(4000):
    k = int(rng() * 100)
    if rng() < 0.5:
        sl2.insert(k, k * 7)
        ref[k] = k * 7
    else:
        sl2.delete(k)
        ref.pop(k, None)
check("skip list keys match a reference dict", sl2.keys() == sorted(ref))
check("skip list length matches reference", len(sl2) == len(ref))
check("every search matches the reference", all(sl2.search(k) == ref.get(k) for k in range(100)))
check("random range query matches reference",
      sl2.range(20, 60) == sorted((k, v) for k, v in ref.items() if 20 <= k <= 60))

# --- ordered traversal is always sorted after arbitrary ops ----------------
keys = sl2.keys()
check("traversal is strictly increasing", all(keys[i] < keys[i + 1] for i in range(len(keys) - 1)))

# --- the level distribution is geometric (about p at each rung) ------------
big = SkipList(p=0.5, seed=11)
for k in range(2000):
    big.insert(k, k)
levels = big.node_levels()
frac_ge1 = sum(1 for l in levels if l >= 1) / len(levels)
frac_ge2 = sum(1 for l in levels if l >= 2) / len(levels)
check("about half the nodes reach level >= 1", 0.4 < frac_ge1 < 0.6)
check("about a quarter reach level >= 2", 0.15 < frac_ge2 < 0.35)
check("all keys present after 2000 inserts", big.keys() == list(range(2000)))

# --- string keys work too --------------------------------------------------
words = SkipList(seed=3)
for w in ["pear", "apple", "cherry", "banana"]:
    words.insert(w, len(w))
check("string keys sort lexicographically",
      words.keys() == ["apple", "banana", "cherry", "pear"])
check("string range query", words.range("apple", "cherry") == [("apple", 5), ("banana", 6), ("cherry", 6)])

# --- determinism for a fixed seed ------------------------------------------
a = SkipList(seed=99)
b = SkipList(seed=99)
for k in [3, 1, 4, 1, 5, 9, 2, 6]:
    a.insert(k, k)
    b.insert(k, k)
check("same seed gives same structure", a.node_levels() == b.node_levels())

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all skiplist tests passed")
