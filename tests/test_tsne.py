"""Tests for tsne: valid P, perplexity calibration, KL decrease, cluster separation, trustworthiness."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tsne import (tsne, joint_probabilities, trustworthiness, _conditional_p_row,
                  _pairwise_sq_dists)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 1
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def make_clusters(centers, per=15, spread=1.0, dim=None):
    X, labels = [], []
    dim = dim or len(centers[0])
    for c, ctr in enumerate(centers):
        for _ in range(per):
            X.append([ctr[k] + (rng() * 2 - 1) * spread for k in range(dim)])
            labels.append(c)
    return X, labels


# --- joint probability matrix is a valid symmetric distribution ------------
X, labels = make_clusters([[0, 0, 0, 0, 0], [10, 10, 0, 0, 0], [0, 0, 10, 10, 0]], per=12)
P = joint_probabilities(X, perplexity=8)
n = len(X)
check("P sums to 1", abs(sum(sum(row) for row in P) - 1.0) < 1e-9)
check("P is symmetric", all(abs(P[i][j] - P[j][i]) < 1e-15 for i in range(n) for j in range(n)))
check("P diagonal is ~0 (no self-similarity mass)", all(P[i][i] < 1e-9 for i in range(n)))
check("P is non-negative", all(P[i][j] >= 0 for i in range(n) for j in range(n)))

# --- perplexity calibration hits the target --------------------------------
# for a single point's row, the achieved perplexity (2^entropy) should match the requested one
D = _pairwise_sq_dists(X)
for target_perp in [5.0, 15.0, 30.0]:
    row = _conditional_p_row(D[0], 0, math.log(target_perp))
    H = -sum(p * math.log(p) for p in row if p > 0)   # entropy in nats
    achieved = math.exp(H)
    check(f"perplexity search hits {target_perp} (got {achieved:.2f})",
          abs(achieved - target_perp) < 0.5)

# --- KL divergence decreases over training ---------------------------------
Y, hist = tsne(X, perplexity=8, n_iter=300, return_history=True, seed=1)
check("KL divergence decreases from start to end", hist[-1] < hist[0])
check("final KL is small", hist[-1] < 1.0)
# roughly monotone: the last quarter should be below the first quarter's mean
q = len(hist) // 4
check("KL trends downward (last quarter below first quarter)",
      sum(hist[-q:]) / q < sum(hist[:q]) / q)

# --- well-separated clusters map to separated 2-D clusters -----------------
def separation_ratio(Y, labels):
    intra, inter = [], []
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            d = math.hypot(Y[i][0] - Y[j][0], Y[i][1] - Y[j][1])
            (intra if labels[i] == labels[j] else inter).append(d)
    return (sum(inter) / len(inter)) / (sum(intra) / len(intra) + 1e-12)

check(f"3 clusters are well separated in the embedding (ratio {separation_ratio(Y, labels):.2f})",
      separation_ratio(Y, labels) > 2.0)

# --- trustworthiness is high -----------------------------------------------
tw = trustworthiness(X, Y, k=5)
check(f"trustworthiness is high ({tw:.3f})", tw > 0.9)

# --- a single blob (no structure) still produces a finite, centered embedding
Xblob, _ = make_clusters([[0, 0, 0]], per=30, spread=1.0)
Yb = tsne(Xblob, perplexity=10, n_iter=150, seed=2)
cx = sum(p[0] for p in Yb) / len(Yb)
cy = sum(p[1] for p in Yb) / len(Yb)
check("embedding is centered near the origin", abs(cx) < 1e-6 and abs(cy) < 1e-6)
check("embedding coordinates are finite", all(math.isfinite(v) for p in Yb for v in p))

# --- two clusters clearly split into two groups ----------------------------
X2, lab2 = make_clusters([[0, 0, 0, 0], [8, 8, 8, 8]], per=15, spread=0.8)
Y2 = tsne(X2, perplexity=8, n_iter=300, seed=3)
check("two clusters separate (ratio > 3)", separation_ratio(Y2, lab2) > 3.0)

# --- reproducibility -------------------------------------------------------
Ya = tsne(X2, perplexity=8, n_iter=100, seed=9)
Yb2 = tsne(X2, perplexity=8, n_iter=100, seed=9)
check("same seed gives identical embedding", Ya == Yb2)

# --- output shape ----------------------------------------------------------
check("embedding has one point per input", len(Y) == len(X))
check("embedding is 2-dimensional by default", all(len(p) == 2 for p in Y))
Y3 = tsne(X2, perplexity=8, n_iter=50, n_components=3, seed=1)
check("n_components=3 gives 3-D points", all(len(p) == 3 for p in Y3))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all tsne tests passed")
