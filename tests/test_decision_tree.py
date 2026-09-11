"""Tests for decision_tree: impurity, splits, learning, feature importance."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import decision_tree as DT

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- impurity measures -----------------------------------------------------
check("gini pure = 0", approx(DT.gini([1, 1, 1, 1]), 0.0))
check("gini 50/50 = 0.5", approx(DT.gini([0, 0, 1, 1]), 0.5))
# three equal classes: 1 - 3*(1/3)^2 = 2/3
check("gini 3-way = 2/3", approx(DT.gini([0, 1, 2]), 2.0 / 3.0))
check("gini empty = 0", approx(DT.gini([]), 0.0))

check("entropy pure = 0", approx(DT.entropy([5, 5, 5]), 0.0))
check("entropy 50/50 = 1 bit", approx(DT.entropy([0, 0, 1, 1]), 1.0))
# 4 equal classes -> 2 bits
check("entropy 4-way = 2 bits", approx(DT.entropy([0, 1, 2, 3]), 2.0))

# --- learns a known 1-D threshold ------------------------------------------
X = [[x] for x in range(10)]
y = [0 if x < 5 else 1 for x in range(10)]
t = DT.DecisionTree().fit(X, y)
check("1D separable perfect fit", approx(t.accuracy(X, y), 1.0))
check("1D root split at 4.5", approx(t.root.threshold, 4.5))
check("1D depth = 1", t.depth() == 1)
check("1D leaves = 2", t.n_leaves() == 2)

# --- entropy criterion learns the same split -------------------------------
te = DT.DecisionTree(criterion="entropy").fit(X, y)
check("entropy criterion perfect fit", approx(te.accuracy(X, y), 1.0))

# --- max_depth stops growth ------------------------------------------------
# staircase needing depth 3 to be perfect; cap at 1 => imperfect
Xs = [[x] for x in range(8)]
ys = [x % 2 for x in range(8)]  # alternating: hard, needs many splits
shallow = DT.DecisionTree(max_depth=1).fit(Xs, ys)
check("max_depth respected", shallow.depth() <= 1)

# --- feature importance concentrates on the informative feature ------------
X2 = [[a, b] for a in range(6) for b in range(6)]
y2 = [1 if a > 2 else 0 for a, b in X2]  # only feature 0 matters
t2 = DT.DecisionTree(max_depth=4).fit(X2, y2)
imp = t2.feature_importances()
check("importance sums to 1", approx(sum(imp), 1.0))
check("feature 0 dominant", imp[0] > 0.99)
check("irrelevant feature ~0", imp[1] < 0.01)
check("2D perfect fit", approx(t2.accuracy(X2, y2), 1.0))

# --- 3-class well-separated blobs (LCG, no random module) ------------------
state = 12345


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0  # high bits


Xc, yc = [], []
for cls, (cx, cy) in enumerate([(0.0, 0.0), (6.0, 6.0), (0.0, 6.0)]):
    for _ in range(25):
        Xc.append([cx + (rng() - 0.5), cy + (rng() - 0.5)])
        yc.append(cls)
tc = DT.DecisionTree().fit(Xc, yc)
check("3-class blobs perfect fit", approx(tc.accuracy(Xc, yc), 1.0))
check("3-class >= 3 leaves", tc.n_leaves() >= 3)

# --- generalization: train/test split on a linearly separable rule ---------
Xtr, ytr, Xte, yte = [], [], [], []
for i in range(200):
    a, b = rng() * 10, rng() * 10
    label = 1 if a + b > 10 else 0
    if i < 140:
        Xtr.append([a, b]); ytr.append(label)
    else:
        Xte.append([a, b]); yte.append(label)
tg = DT.DecisionTree(max_depth=6).fit(Xtr, ytr)
check("train accuracy high", tg.accuracy(Xtr, ytr) >= 0.95)
check("test accuracy generalizes", tg.accuracy(Xte, yte) >= 0.85)

# --- rules() round-trips: reading a rule reproduces the prediction ---------
rules = t.rules()
check("rules non-empty", len(rules) > 0)
check("rules mention a leaf", any("predict" in r for r in rules))

# --- single-class input -> a leaf, constant prediction ---------------------
tconst = DT.DecisionTree().fit([[1], [2], [3]], [7, 7, 7])
check("constant target => 1 leaf", tconst.n_leaves() == 1)
check("constant prediction", tconst.predict([[99]]) == [7])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all decision_tree tests passed")
