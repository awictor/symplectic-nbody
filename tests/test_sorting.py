"""Tests for sorting: correctness vs sorted(), stability, heap, comparison-count complexity."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sorting import (insertion_sort, merge_sort, quick_sort, heap_sort, BinaryHeap,
                     is_sorted, Counter, ALGORITHMS, STABLE)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 5


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- correctness on a range of input shapes --------------------------------
inputs = {
    "random": [int(rng() * 100) for _ in range(200)],
    "sorted": list(range(100)),
    "reverse": list(range(100, 0, -1)),
    "duplicates": [int(rng() * 5) for _ in range(200)],
    "empty": [],
    "single": [42],
    "two": [2, 1],
    "all-equal": [7] * 30,
}
for name, alg in ALGORITHMS.items():
    ok = all(alg(inp) == sorted(inp) for inp in inputs.values())
    check(f"{name} sort matches sorted() on all inputs", ok)

# --- does not mutate the input ---------------------------------------------
original = [3, 1, 2]
for name, alg in ALGORITHMS.items():
    src = list(original)
    alg(src)
    check(f"{name} does not mutate its input", src == original)

# --- key function support --------------------------------------------------
words = ["bbb", "a", "cc", "dddd"]
for name, alg in ALGORITHMS.items():
    check(f"{name} sorts by key", alg(words, key=len) == ["a", "cc", "bbb", "dddd"])

# --- stability: equal keys keep their original relative order --------------
pairs = [(int(rng() * 3), i) for i in range(60)]
for name, alg in ALGORITHMS.items():
    s = alg(pairs, key=lambda p: p[0])
    stable = all(s[i][1] < s[i + 1][1] for i in range(len(s) - 1) if s[i][0] == s[i + 1][0])
    check(f"{name} stability = {name in STABLE}", stable == (name in STABLE))

# --- is_sorted helper ------------------------------------------------------
check("is_sorted true for sorted", is_sorted([1, 2, 2, 3]))
check("is_sorted false for unsorted", not is_sorted([1, 3, 2]))
check("is_sorted with key", is_sorted(["a", "bb", "ccc"], key=len))

# --- binary heap: validity and priority-queue behaviour --------------------
h = BinaryHeap([5, 3, 8, 1, 9, 2, 7])
check("heapify produces a valid heap", h.is_valid())
check("peek is the minimum", h.peek() == 1)
popped = [h.pop() for _ in range(7)]
check("heap pops in sorted order", popped == sorted([5, 3, 8, 1, 9, 2, 7]))
check("emptied heap has length 0", len(h) == 0)

# --- heap push maintains the invariant -------------------------------------
h2 = BinaryHeap()
for v in [4, 1, 7, 3, 9, 2, 8, 5]:
    h2.push(v)
    if not h2.is_valid():
        check("heap valid after each push", False)
        break
else:
    check("heap valid after each push", True)
check("push/pop yields sorted stream", [h2.pop() for _ in range(len(h2))] == [1, 2, 3, 4, 5, 7, 8, 9])

# --- empty heap raises -----------------------------------------------------
def raises(fn):
    try:
        fn()
        return False
    except IndexError:
        return True


check("pop from empty heap raises", raises(lambda: BinaryHeap().pop()))
check("peek at empty heap raises", raises(lambda: BinaryHeap().peek()))

# --- heap with a key function ----------------------------------------------
hk = BinaryHeap(["ccc", "a", "bb"], key=len)
check("heap min by key", hk.peek() == "a")

# --- comparison complexity: O(n log n) sorts scale far better than O(n^2) --
n = 600
big = [int(rng() * 100000) for _ in range(n)]
counts = {}
for name, alg in ALGORITHMS.items():
    c = Counter()
    alg(big, counter=c)
    counts[name] = c.comparisons
nlogn = n * math.log2(n)
check("merge sort is O(n log n)", counts["merge"] < 5 * nlogn)
check("heap sort is O(n log n)", counts["heap"] < 5 * nlogn)
check("quick sort is O(n log n) on random data", counts["quick"] < 8 * nlogn)
check("insertion sort is O(n^2) (far more comparisons)", counts["insertion"] > 8 * nlogn)

# --- quick sort survives its classic adversaries (sorted / reverse) --------
# median-of-three keeps these O(n log n), not O(n^2)
srt = list(range(2000))
c_sorted = Counter()
quick_sort(srt, counter=c_sorted)
check("quick sort handles pre-sorted input efficiently",
      c_sorted.comparisons < 40 * 2000 and quick_sort(srt) == srt)
rev = list(range(2000, 0, -1))
c_rev = Counter()
res_rev = quick_sort(rev, counter=c_rev)
check("quick sort handles reverse input efficiently",
      c_rev.comparisons < 40 * 2000 and res_rev == sorted(rev))

# --- a big random sort is exactly correct ----------------------------------
huge = [int(rng() * 1000000) for _ in range(3000)]
ref = sorted(huge)
for name, alg in ALGORITHMS.items():
    if name == "insertion":
        continue     # skip the O(n^2) one at this size
    check(f"{name} correct on 3000 random elements", alg(huge) == ref)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all sorting tests passed")
