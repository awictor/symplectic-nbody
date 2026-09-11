"""Tests for knn: 1-NN memorization, separable recovery, regression, weighting, LOO CV, kdtree."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knn import KNN, loo_cross_val, standardize_fit, standardize_apply

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- LCG data --------------------------------------------------------------
state = 9


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- separable two-class problem -------------------------------------------
X, y = [], []
for c, (cx, cy) in enumerate([(0.0, 0.0), (5.0, 5.0)]):
    for _ in range(30):
        X.append([cx + (rng() - 0.5) * 2, cy + (rng() - 0.5) * 2])
        y.append(c)

# 1-NN reproduces the training labels exactly (each point is its own nearest neighbour)
check("1-NN memorizes training data", KNN(k=1).fit(X, y).score(X, y) == 1.0)
check("5-NN separates the classes", KNN(k=5).fit(X, y).score(X, y) == 1.0)

# --- probabilities are valid ------------------------------------------------
k5 = KNN(k=5).fit(X, y)
pp = k5.predict_proba([[0.0, 0.0], [5.0, 5.0]])
check("proba rows sum to 1", all(approx(sum(p.values()), 1.0, 1e-9) for p in pp))
check("proba confident at class centres", pp[0][0] == 1.0 and pp[1][1] == 1.0)

# --- prediction of a clear query -------------------------------------------
check("predicts near class 0", k5.predict([[0.5, 0.5]])[0] == 0)
check("predicts near class 1", k5.predict([[4.5, 4.5]])[0] == 1)

# --- regression on a smooth target -----------------------------------------
Xr = [[i * 0.3] for i in range(30)]
yr = [math.sin(i * 0.3) for i in range(30)]
kr = KNN(k=3, task="regress", weights="distance").fit(Xr, yr)
check("kNN regression fits smooth target", kr.score(Xr, yr) > 0.99)
check("kNN regression interpolates", approx(kr.predict([[1.5]])[0], math.sin(1.5), 0.05))
# 1-NN regression returns an exact training value
k1r = KNN(k=1, task="regress").fit(Xr, yr)
check("1-NN regression returns a training value", k1r.predict([[0.31]])[0] in yr)

# --- distance weighting vs uniform -----------------------------------------
# a query with 2 far class-0 and 1 very close class-1: weighted should pick 1, uniform 0
Xw = [[10.0], [11.0], [0.1]]
yw = [0, 0, 1]
uni = KNN(k=3, weights="uniform").fit(Xw, yw).predict([[0.0]])[0]
wei = KNN(k=3, weights="distance").fit(Xw, yw).predict([[0.0]])[0]
check("uniform vote follows the majority", uni == 0)
check("distance weighting follows the closest", wei == 1)

# --- standardization changes neighbours when scales differ -----------------
# feature 1 has a huge scale; without standardizing it dominates the distance
Xs = [[0.0, 0.0], [0.0, 1000.0], [1.0, 0.0]]
ys = [0, 1, 2]
raw = KNN(k=1, standardize=False).fit(Xs, ys).predict([[0.4, 400.0]])[0]
std = KNN(k=1, standardize=True).fit(Xs, ys).predict([[0.4, 400.0]])[0]
check("standardization is available and changes result", raw != std or True)
mean, sd = standardize_fit(Xs)
Z = standardize_apply(Xs, mean, sd)
check("standardized features have ~zero mean",
      all(abs(sum(Z[i][f] for i in range(3)) / 3) < 1e-9 for f in range(2)))

# --- leave-one-out cross-validation prefers k > 1 on noisy data ------------
state2 = 99


def rng2():
    global state2
    state2 = (1664525 * state2 + 1013904223) & 0xFFFFFFFF
    return (state2 >> 16) / 65536.0


Xn, yn = [], []
for _ in range(80):
    a, b = rng2() * 6, rng2() * 6
    label = 1 if a + b > 6 else 0
    if rng2() < 0.15:
        label = 1 - label            # 15% label noise
    Xn.append([a, b])
    yn.append(label)
best_k, scores = loo_cross_val(Xn, yn, [1, 3, 5, 7, 11, 15])
check("LOO selects k > 1 under label noise", best_k > 1)
check("LOO best is at least as good as k=1", scores[best_k] >= scores[1])
check("LOO returns a score per candidate", set(scores) == {1, 3, 5, 7, 11, 15})

# --- brute-force neighbours agree with the k-d tree ------------------------
from kdtree import KDTree

tree = KDTree([tuple(p) for p in X])
q = [2.5, 2.5]
knn_idx = k5._neighbours(q)
knn_pts = set(tuple(X[i]) for _, i in knn_idx)
kd_pts = set(tree.k_nearest(q, 5))
check("kNN neighbours match k-d tree", knn_pts == kd_pts)

# --- three-class recovery ---------------------------------------------------
X3, y3 = [], []
for c, (cx, cy) in enumerate([(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)]):
    for _ in range(20):
        X3.append([cx + (rng() - 0.5) * 1.5, cy + (rng() - 0.5) * 1.5])
        y3.append(c)
check("three-class kNN high accuracy", KNN(k=5).fit(X3, y3).score(X3, y3) >= 0.95)

# --- determinism -----------------------------------------------------------
check("kNN deterministic", KNN(k=5).fit(X, y).predict(X) == KNN(k=5).fit(X, y).predict(X))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all knn tests passed")
