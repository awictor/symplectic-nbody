"""Tests for spectral_clustering: non-convex shapes, Laplacian spectrum, component counting."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectral_clustering import (spectral_clustering, affinity_matrix, laplacian,
                                 smallest_eigenvectors, count_components, knn_affinity)
from kmeans import kmeans_best

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


state = 5


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def purity(labels, truth, n_cls):
    for c in set(labels):
        ts = [truth[i] for i in range(len(labels)) if labels[i] == c]
        if max(ts.count(t) for t in range(n_cls)) / len(ts) < 0.9:
            return False
    return True


# --- affinity matrix properties -------------------------------------------
X = [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]]
W = affinity_matrix(X, sigma=1.0)
check("affinity symmetric", all(approx(W[i][j], W[j][i], 1e-12) for i in range(3) for j in range(3)))
check("affinity zero self", all(W[i][i] == 0.0 for i in range(3)))
check("near points high affinity", W[0][1] > 0.9)
check("far points low affinity", W[0][2] < 1e-6)

# --- Laplacian structure ---------------------------------------------------
Lun = laplacian(W, normalized=False)
# unnormalized Laplacian rows sum to zero
check("unnormalized Laplacian rows sum to 0",
      all(approx(sum(Lun[i]), 0.0, 1e-9) for i in range(3)))
Lsym = laplacian(W, normalized=True)
check("normalized Laplacian diagonal ~1",
      all(approx(Lsym[i][i], 1.0, 1e-9) for i in range(3) if sum(W[i]) > 1e-12))

# --- smallest eigenvalue of a Laplacian is ~0 (constant eigenvector) -------
# a single connected blob -> exactly one zero eigenvalue
blob = [[rng(), rng()] for _ in range(12)]
Wb = affinity_matrix(blob, sigma=2.0)
Lb = laplacian(Wb, normalized=True)
vals, vecs = smallest_eigenvectors(Lb, 3)
check("smallest Laplacian eigenvalue ~ 0", approx(vals[0], 0.0, 1e-4))
check("eigenvalues sorted ascending", vals[0] <= vals[1] <= vals[2] + 1e-9)

# --- zero-eigenvalue multiplicity = number of connected components ---------
# three far-apart blobs => 3 components, 3 near-zero eigenvalues
comp = []
for cx, cy in [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)]:
    for _ in range(6):
        comp.append([cx + (rng() - 0.5), cy + (rng() - 0.5)])
Wc = affinity_matrix(comp, sigma=1.0)
check("union-find counts 3 components", count_components(Wc, tol=1e-3) == 3)
Lc = laplacian(Wc, normalized=True)
vc, _ = smallest_eigenvectors(Lc, 4)
n_zero = sum(1 for v in vc if v < 1e-3)
check("3 near-zero Laplacian eigenvalues", n_zero == 3)
check("4th eigenvalue clearly positive (spectral gap)", vc[3] > 0.1)

# --- two concentric rings: spectral succeeds, k-means fails ----------------
# a dedicated RNG so this data is independent of test ordering above
_rs = [42]


def _rrng():
    _rs[0] = (1664525 * _rs[0] + 1013904223) & 0xFFFFFFFF
    return (_rs[0] >> 16) / 65536.0


def rings(n):
    Xr, yr = [], []
    for _ in range(n):
        t = _rrng() * 2 * math.pi
        Xr.append([math.cos(t) + (_rrng() - 0.5) * 0.08, math.sin(t) + (_rrng() - 0.5) * 0.08])
        yr.append(0)
        Xr.append([3 * math.cos(t) + (_rrng() - 0.5) * 0.08,
                   3 * math.sin(t) + (_rrng() - 0.5) * 0.08])
        yr.append(1)
    return Xr, yr


Xr, yr = rings(18)     # 36 points, keeps the O(n^3) eigensolve fast
lab, vals = spectral_clustering(Xr, 2, sigma=0.4, seed=1)
check("spectral separates concentric rings", purity(lab, yr, 2))
_, kmlab, _ = kmeans_best(Xr, 2, seed=1)
check("k-means fails on the same rings", not purity(kmlab, yr, 2))
check("rings: two near-zero eigenvalues", vals[0] < 1e-2 and vals[1] < 0.1)

# --- two interlocking moons ------------------------------------------------
def moons(n):
    Xm, ym = [], []
    for _ in range(n):
        t = rng() * math.pi
        Xm.append([math.cos(t) + (rng() - 0.5) * 0.1, math.sin(t) + (rng() - 0.5) * 0.1])
        ym.append(0)
        Xm.append([1 - math.cos(t) + (rng() - 0.5) * 0.1,
                   -math.sin(t) + 0.4 + (rng() - 0.5) * 0.1])
        ym.append(1)
    return Xm, ym


Xm, ym = moons(25)
lm, _ = spectral_clustering(Xm, 2, sigma=0.3, seed=2)
check("spectral separates two moons", purity(lm, ym, 2))

# --- plain blobs still work ------------------------------------------------
Xb, yb = [], []
for c, (cx, cy) in enumerate([(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)]):
    for _ in range(10):
        Xb.append([cx + (rng() - 0.5), cy + (rng() - 0.5)])
        yb.append(c)
lb, _ = spectral_clustering(Xb, 3, sigma=1.0, seed=3)
check("spectral recovers plain blobs", purity(lb, yb, 3))
check("spectral gives 3 clusters", len(set(lb)) == 3)

# --- labels cover all points, contiguous ----------------------------------
check("every point labeled", len(lm) == len(Xm))
check("labels contiguous", sorted(set(lb)) == list(range(len(set(lb)))))

# --- knn affinity is symmetric and sparser --------------------------------
Wk = knn_affinity(Xb, k=5, sigma=1.0)
check("knn affinity symmetric", all(approx(Wk[i][j], Wk[j][i], 1e-12)
                                    for i in range(len(Xb)) for j in range(len(Xb))))
nz_full = sum(1 for i in range(len(Xb)) for j in range(len(Xb)) if affinity_matrix(Xb, 1.0)[i][j] > 1e-6)
nz_knn = sum(1 for i in range(len(Xb)) for j in range(len(Xb)) if Wk[i][j] > 1e-6)
check("knn affinity is sparser than full", nz_knn <= nz_full)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all spectral_clustering tests passed")
