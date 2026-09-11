"""Tests for kmeans.py -- k-means clustering.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Verifies blob recovery, monotone
inertia, k-means++ seeding, and the silhouette separation score.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import kmeans as KM  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def _blob(cx, cy, n, spread, seed):
    r = KM._Rng(seed)
    return [[cx + (r.random() - 0.5) * spread, cy + (r.random() - 0.5) * spread] for _ in range(n)]


# three well-separated blobs
data = _blob(0, 0, 30, 1, 1) + _blob(10, 0, 30, 1, 2) + _blob(5, 9, 30, 1, 3)

# --- recovers well-separated blobs -----------------------------------------
c, labels, inertia, it = KM.kmeans(data, 3, seed=1)
check("produces k centroids", len(c) == 3)
check("assigns a label to every point", len(labels) == len(data))
check("all three clusters are non-empty", all(sz > 0 for sz in KM.cluster_sizes(labels, 3)))
l1, l2, l3 = set(labels[:30]), set(labels[30:60]), set(labels[60:])
check("each blob lands in a single cluster", len(l1) == 1 and len(l2) == 1 and len(l3) == 1)
check("the three blobs map to three distinct clusters", len({*l1, *l2, *l3}) == 3)
check("cluster sizes are 30/30/30", sorted(KM.cluster_sizes(labels, 3)) == [30, 30, 30])
# centroids near the true centres
centres = sorted(tuple(round(x, 0) for x in ct) for ct in c)
check("centroids recover the true blob centres", centres == [(0.0, 0.0), (5.0, 9.0), (10.0, 0.0)])

# --- inertia -----------------------------------------------------------------
check("inertia matches a direct recomputation", abs(inertia - KM.inertia(data, c, labels)) < 1e-9)
check("inertia is positive for spread-out data", inertia > 0)
# inertia is non-increasing as more iterations run
prev = None
mono = True
for mi in range(1, 8):
    _, _, inr, _ = KM.kmeans(data, 3, seed=1, max_iter=mi)
    if prev is not None and inr > prev + 1e-9:
        mono = False
    prev = inr
check("inertia decreases monotonically across iterations", mono)
# more clusters -> lower (or equal) inertia
check("more clusters lower the inertia",
      KM.kmeans(data, 5, seed=1)[2] <= KM.kmeans(data, 3, seed=1)[2] + 1e-9)

# --- silhouette --------------------------------------------------------------
check("well-separated blobs have a high silhouette", KM.silhouette(data, labels, 3) > 0.7)
# a bad clustering (k=2 splitting the wrong way) scores lower than the true k=3
sil3 = KM.silhouette(data, KM.kmeans(data, 3, seed=1)[1], 3)
sil2 = KM.silhouette(data, KM.kmeans(data, 2, seed=1)[1], 2)
check("the natural k=3 clustering out-scores k=2", sil3 > sil2)

# --- k-means++ seeding ------------------------------------------------------
rng = KM._Rng(1)
seeds = KM.init_plus_plus(data, 3, rng)
check("k-means++ picks k seeds", len(seeds) == 3)
check("k-means++ seeds are spread out (not all in one blob)",
      max(KM._dist2(seeds[i], seeds[j]) for i in range(3) for j in range(3)) > 25)
# random init also works, just less reliably
cr, lr, ir, _ = KM.kmeans(data, 3, init="random", seed=5)
check("random init also produces a valid clustering", len(set(lr)) <= 3 and ir > 0)

# --- multi-restart selection ------------------------------------------------
cb, lb, ib = KM.kmeans_best(data, 3, restarts=8)
check("kmeans_best returns the lowest inertia found",
      ib <= KM.kmeans(data, 3, seed=1)[2] + 1e-9)
check("best clustering also recovers the blobs", sorted(KM.cluster_sizes(lb, 3)) == [30, 30, 30])

# --- edge cases -------------------------------------------------------------
check("k=1 puts everything in one cluster", KM.cluster_sizes(KM.kmeans(data, 1, seed=1)[1], 1) == [len(data)])
check("k = n gives zero inertia (each point its own cluster)",
      abs(KM.kmeans([[0], [1], [2]], 3, seed=1)[2]) < 1e-9)
check("identical points cluster with zero inertia", abs(KM.kmeans([[5, 5]] * 10, 1, seed=1)[2]) < 1e-9)
try:
    KM.kmeans(data, 0)
    check("rejects k <= 0", False)
except ValueError:
    check("rejects k <= 0", True)
try:
    KM.kmeans([[1], [2]], 5)
    check("rejects k > n", False)
except ValueError:
    check("rejects k > n", True)

# --- 3D data ----------------------------------------------------------------
r3 = KM._Rng(42)
d3 = []
for cx, cy, cz in ((0.0, 0.0, 0.0), (8.0, 8.0, 8.0)):
    for _ in range(20):
        d3.append([cx + (r3.random() - 0.5), cy + (r3.random() - 0.5), cz + (r3.random() - 0.5)])
c3, l3d, i3, _ = KM.kmeans(d3, 2, seed=1)
check("k-means works in 3D", sorted(KM.cluster_sizes(l3d, 2)) == [20, 20])


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall kmeans tests passed")
