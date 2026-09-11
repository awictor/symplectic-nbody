"""Tests for random_forest: bagging, OOB scoring, ensemble beats overfit tree."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from random_forest import RandomForest, _majority_vote
from decision_tree import DecisionTree

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- majority vote ---------------------------------------------------------
check("majority vote basic", _majority_vote([1, 1, 0]) == 1)
check("majority vote tie -> larger key", _majority_vote([0, 1]) == 1)

# --- LCG data generator (high bits) ----------------------------------------
state = 999


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def make_data(n, noise_features=4, flip=0.10):
    X, y = [], []
    for _ in range(n):
        a, b = rng() * 10, rng() * 10
        extra = [rng() * 10 for _ in range(noise_features)]
        label = 1 if (a - 5) ** 2 + (b - 5) ** 2 < 9 else 0
        if rng() < flip:
            label = 1 - label
        X.append([a, b] + extra)
        y.append(label)
    return X, y


X, y = make_data(500)
Xtr, ytr, Xte, yte = X[:350], y[:350], X[350:], y[350:]

rf = RandomForest(n_trees=41, max_depth=8, seed=1).fit(Xtr, ytr)
dt = DecisionTree().fit(Xtr, ytr)  # unlimited depth -> memorizes noise

# --- basic structure -------------------------------------------------------
check("built n_trees", len(rf.trees) == 41)
check("every tree has an oob set", all(len(o) > 0 for o in rf.oob_indices))
check("predict length matches", len(rf.predict(Xte)) == len(Xte))

# --- ensemble generalizes better than a single overfit tree ---------------
dt_train = dt.accuracy(Xtr, ytr)
dt_test = dt.accuracy(Xte, yte)
rf_test = rf.accuracy(Xte, yte)
check("single tree overfits (train ~ 1.0)", dt_train > 0.99)
check("forest beats overfit tree on test", rf_test >= dt_test)
check("forest test accuracy reasonable", rf_test >= 0.80)

# --- out-of-bag score is a valid free estimate that tracks test error ------
check("oob score computed", rf.oob_score_ is not None)
check("oob tracks test error", abs(rf.oob_score_ - rf_test) < 0.06)

# --- feature importance: real features (0,1) dominate the noise features ---
imp = rf.feature_importances()
check("importances sum to 1", abs(sum(imp) - 1.0) < 1e-9)
check("real features dominate noise", (imp[0] + imp[1]) > sum(imp[2:]))

# --- perfectly separable data -> 100% ------------------------------------
Xs = [[i] for i in range(12)]
ys = [0 if i < 6 else 1 for i in range(12)]
rfs = RandomForest(n_trees=15, max_features=None, seed=3).fit(Xs, ys)
check("separable forest perfect", rfs.accuracy(Xs, ys) == 1.0)

# --- more trees do not hurt (monotone-ish stability) ----------------------
small = RandomForest(n_trees=5, max_depth=8, seed=1).fit(Xtr, ytr)
big = RandomForest(n_trees=61, max_depth=8, seed=1).fit(Xtr, ytr)
check("more trees no worse than few", big.accuracy(Xte, yte) >= small.accuracy(Xte, yte) - 0.05)

# --- reproducible with same seed ------------------------------------------
a = RandomForest(n_trees=10, seed=7).fit(Xtr, ytr).predict(Xte)
b = RandomForest(n_trees=10, seed=7).fit(Xtr, ytr).predict(Xte)
check("same seed reproducible", a == b)

# --- 3-class blobs ---------------------------------------------------------
Xc, yc = [], []
for cls, (cx, cy) in enumerate([(0.0, 0.0), (8.0, 8.0), (0.0, 8.0)]):
    for _ in range(40):
        Xc.append([cx + (rng() - 0.5) * 3, cy + (rng() - 0.5) * 3])
        yc.append(cls)
rfc = RandomForest(n_trees=25, seed=4).fit(Xc, yc)
check("3-class blobs high accuracy", rfc.accuracy(Xc, yc) >= 0.95)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all random_forest tests passed")
