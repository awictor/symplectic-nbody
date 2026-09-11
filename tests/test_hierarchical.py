"""Tests for hierarchical: dendrogram structure, monotone merges, linkage behaviour, cutting."""

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hierarchical import (linkage, fcluster, merge_heights, is_monotone, LINKAGES)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- LCG data --------------------------------------------------------------
state = 7


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# three well-separated blobs
X = []
truth = []
for c, (cx, cy) in enumerate([(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)]):
    for _ in range(15):
        X.append([cx + (rng() - 0.5), cy + (rng() - 0.5)])
        truth.append(c)


def purity(labels):
    ok = True
    for cl in set(labels):
        idx = [i for i in range(len(labels)) if labels[i] == cl]
        best = max(sum(1 for i in idx if truth[i] == t) for t in range(3))
        if best / len(idx) < 0.9:
            ok = False
    return ok


# --- every linkage builds n-1 merges, monotone, and recovers the blobs -----
for method in LINKAGES:
    merges = linkage(X, method)
    check(f"{method}: n-1 merges", len(merges) == len(X) - 1)
    check(f"{method}: merge heights monotone", is_monotone(merges))
    labels = fcluster(X, 3, method)
    check(f"{method}: three clusters", len(set(labels)) == 3)
    check(f"{method}: clusters are pure", purity(labels))

# --- merge heights strictly grow from small to large ----------------------
h = merge_heights(linkage(X, "ward"))
check("ward heights increase overall", h[-1] > h[0])
check("ward first merge is small", h[0] < 1.0)

# --- cutting the tree: k clusters means k labels ---------------------------
for k in (1, 2, 3, 5, 10):
    labels = fcluster(X, k, "average")
    check(f"cut into {k} gives {k} labels", len(set(labels)) == k)
check("k >= n gives every point its own cluster", len(set(fcluster(X, len(X) + 5, "single"))) == len(X))
check("k = 1 gives one cluster", len(set(fcluster(X, 1, "complete"))) == 1)

# --- single vs complete linkage on a long chain ---------------------------
# single linkage chains (peels a small end group); complete stays compact (even split)
chain = [[i * 0.3, 0.0] for i in range(24)]
single_sizes = sorted(Counter(fcluster(chain, 2, "single")).values())
complete_sizes = sorted(Counter(fcluster(chain, 2, "complete")).values())
check("single linkage splits unevenly (chaining)", single_sizes[0] < complete_sizes[0])
check("complete linkage splits more evenly", complete_sizes[0] >= single_sizes[0])
check("both split the chain into two", len(single_sizes) == 2 and len(complete_sizes) == 2)

# --- two clearly separated clusters: all linkages agree --------------------
two = [[0.0, 0.0], [0.3, 0.1], [0.1, 0.2], [10.0, 10.0], [10.2, 9.9], [9.8, 10.1]]
for method in LINKAGES:
    lab = fcluster(two, 2, method)
    check(f"{method}: separates two far groups",
          lab[0] == lab[1] == lab[2] and lab[3] == lab[4] == lab[5] and lab[0] != lab[3])

# --- labels are contiguous from 0 ------------------------------------------
lab = fcluster(X, 4, "ward")
check("labels contiguous from 0", sorted(set(lab)) == list(range(4)))
check("every point labeled", len(lab) == len(X))

# --- determinism -----------------------------------------------------------
check("linkage deterministic", linkage(X, "ward") == linkage(X, "ward"))
check("fcluster deterministic", fcluster(X, 3, "ward") == fcluster(X, 3, "ward"))

# --- unknown linkage raises ------------------------------------------------
try:
    linkage(X, "nonsense")
    check("unknown linkage raises", False)
except ValueError:
    check("unknown linkage raises", True)

# --- a hand-checkable 3-point case -----------------------------------------
# points at 0, 1, 5 on a line: first merge joins 0 and 1 (distance 1)
line = [[0.0], [1.0], [5.0]]
m = linkage(line, "single")
check("nearest pair merges first", abs(m[0][2] - 1.0) < 1e-9)
check("second merge is higher", m[1][2] > m[0][2])
check("3-point tree has 2 merges", len(m) == 2)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all hierarchical tests passed")
