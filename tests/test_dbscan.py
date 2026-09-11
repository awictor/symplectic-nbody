"""Tests for dbscan: arbitrary-shape clusters, noise flagging, core/border/noise, auto k."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dbscan import dbscan, classify_points, n_clusters, k_distances, NOISE

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- LCG data helper -------------------------------------------------------
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- two well-separated blobs are found as two clusters --------------------
blobs = []
truth = []
for cx, cy in [(0.0, 0.0), (5.0, 5.0)]:
    for _ in range(30):
        blobs.append([cx + (rng() - 0.5), cy + (rng() - 0.5)])
        truth.append(0 if cx == 0.0 else 1)
lab = dbscan(blobs, eps=0.6, min_pts=4)
check("two blobs -> two clusters", n_clusters(lab) == 2)
check("blobs: no point misassigned across the gap",
      len({lab[i] for i in range(30) if lab[i] != NOISE}) == 1
      and len({lab[i] for i in range(30, 60) if lab[i] != NOISE}) == 1)

# --- two interlocking half-moons (the case k-means fails on) ---------------
def moons(n):
    X, y = [], []
    for _ in range(n):
        t = rng() * math.pi
        X.append([math.cos(t) + (rng() - 0.5) * 0.15,
                  math.sin(t) + (rng() - 0.5) * 0.15])
        y.append(0)
        X.append([1 - math.cos(t) + (rng() - 0.5) * 0.15,
                  -math.sin(t) + 0.4 + (rng() - 0.5) * 0.15])
        y.append(1)
    return X, y


Xm, ym = moons(100)
lm = dbscan(Xm, eps=0.22, min_pts=5)
check("two moons -> two clusters", n_clusters(lm) == 2)
# purity: each discovered cluster is >90% one true label
pure = True
for c in set(lm):
    if c == NOISE:
        continue
    ts = [ym[i] for i in range(len(lm)) if lm[i] == c]
    if max(ts.count(0), ts.count(1)) / len(ts) < 0.9:
        pure = False
check("moon clusters are pure (arbitrary shape recovered)", pure)

# --- noise: clear outliers are labeled NOISE, the dense core is one cluster -
pts = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [5.0, 5.0], [10.0, -10.0]]
ln = dbscan(pts, eps=0.3, min_pts=3)
check("isolated outliers are noise", ln[4] == NOISE and ln[5] == NOISE)
check("dense core forms one cluster", ln[0] == ln[1] == ln[2] == ln[3] != NOISE)

# --- core / border / noise classification ----------------------------------
# a line of 3 close points + 1 far: with min_pts=3, ends are border, middle core, far is noise
line = [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [9.0, 9.0]]
kinds = classify_points(line, eps=0.5, min_pts=3)
check("middle of a dense run is core", kinds[1] == "core")
check("far isolated point is noise", kinds[3] == "noise")
check("classification covers all points", len(kinds) == 4 and set(kinds) <= {"core", "border", "noise"})

# --- number of clusters is discovered, not supplied ------------------------
three = []
for cx, cy in [(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)]:
    for _ in range(25):
        three.append([cx + (rng() - 0.5) * 0.8, cy + (rng() - 0.5) * 0.8])
l3 = dbscan(three, eps=0.7, min_pts=4)
check("discovers three clusters automatically", n_clusters(l3) == 3)

# --- every point is labeled, labels are contiguous from 0 -----------------
check("every point labeled", all(v is not None for v in l3))
present = sorted({v for v in l3 if v != NOISE})
check("cluster ids are contiguous from 0", present == list(range(len(present))))

# --- extreme parameters degrade sensibly -----------------------------------
# huge eps -> everything one cluster; tiny eps -> everything noise
allone = dbscan(three, eps=100.0, min_pts=4)
check("huge eps merges all into one cluster", n_clusters(allone) == 1)
allnoise = dbscan(three, eps=1e-6, min_pts=4)
check("tiny eps makes everything noise", all(v == NOISE for v in allnoise))
# min_pts = 1 makes every point core (no noise possible)
nonoise = dbscan(pts, eps=0.3, min_pts=1)
check("min_pts=1 leaves no noise", NOISE not in nonoise)

# --- k-distance graph is sorted ascending and positive ---------------------
kd = k_distances(Xm, 4)
check("k-distances sorted ascending", kd == sorted(kd))
check("k-distances positive", all(d > 0 for d in kd))
check("k-distances one per point", len(kd) == len(Xm))

# --- determinism: same input, same labels ----------------------------------
check("dbscan is deterministic", dbscan(Xm, 0.22, 5) == dbscan(Xm, 0.22, 5))

# --- empty input -----------------------------------------------------------
check("empty input -> empty labels", dbscan([], 0.5, 3) == [])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all dbscan tests passed")
