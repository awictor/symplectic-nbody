"""Tests for gradient_boosting: monotone loss, beats single tree, learning rate, classification."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gradient_boosting import (GradientBoostingRegressor, GradientBoostingClassifier,
                               _RegTree, _sigmoid)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- sigmoid ---------------------------------------------------------------
check("sigmoid(0) = 0.5", approx(_sigmoid(0.0), 0.5, 1e-12))
check("sigmoid saturates", _sigmoid(40.0) > 0.999 and _sigmoid(-40.0) < 0.001)
check("sigmoid stable for large negative", 0.0 <= _sigmoid(-1000.0) <= 1e-9)

# --- regression tree fits a constant on constant target -------------------
tconst = _RegTree(max_depth=3).fit([[0.0], [1.0], [2.0]], [5.0, 5.0, 5.0])
check("reg tree constant target", approx(tconst.predict_one([9.0]), 5.0, 1e-12))
# a stump splits a step function
tstep = _RegTree(max_depth=1).fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 1.0, 1.0])
check("reg stump splits a step", approx(tstep.predict_one([0.5]), 0.0, 1e-9)
      and approx(tstep.predict_one([2.5]), 1.0, 1e-9))

# --- LCG data --------------------------------------------------------------
state = 17


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- regression on a noisy nonlinear target --------------------------------
X = [[i * 0.1] for i in range(80)]
y = [math.sin(i * 0.1) + 0.3 * (i * 0.1) + (rng() - 0.5) * 0.2 for i in range(80)]
gbr = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1, max_depth=3).fit(X, y)
check("GBR achieves high R^2", gbr.r_squared(X, y) > 0.95)

# --- training loss decreases monotonically as trees are added --------------
staged = gbr.staged_predict(X)
losses = [sum((y[i] - p[i]) ** 2 for i in range(len(y))) / len(y) for p in staged]
check("training loss monotone non-increasing",
      all(losses[i + 1] <= losses[i] + 1e-9 for i in range(len(losses) - 1)))
check("loss falls substantially", losses[-1] < 0.1 * losses[0])
check("staged predictions one per tree", len(staged) == 80)

# --- the ensemble beats a single tree of the same depth --------------------
single = _RegTree(max_depth=3).fit(X, y)
sp = single.predict(X)
single_mse = sum((y[i] - sp[i]) ** 2 for i in range(len(y))) / len(y)
check("boosting beats a single tree", gbr.mse(X, y) < single_mse)

# --- a smaller learning rate needs more trees (underfits at fixed count) ---
lo = GradientBoostingRegressor(n_estimators=15, learning_rate=0.03, max_depth=3).fit(X, y)
hi = GradientBoostingRegressor(n_estimators=15, learning_rate=0.4, max_depth=3).fit(X, y)
check("small learning rate underfits at few trees", lo.mse(X, y) > hi.mse(X, y))

# --- recovers a clean function nearly exactly ------------------------------
Xc = [[i * 0.2] for i in range(40)]
yc = [math.sin(i * 0.2) for i in range(40)]
gclean = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=3).fit(Xc, yc)
check("recovers a clean sine", gclean.r_squared(Xc, yc) > 0.999)

# --- binary classification on a circular boundary --------------------------
Xk, yk = [], []
for _ in range(160):
    a, b = rng() * 4 - 2, rng() * 4 - 2
    yk.append(1 if a * a + b * b < 1.5 else 0)
    Xk.append([a, b])
gbc = GradientBoostingClassifier(n_estimators=60, learning_rate=0.2, max_depth=3).fit(Xk, yk)
check("GBC separates a circular boundary", gbc.accuracy(Xk, yk) >= 0.95)

# --- classification probabilities are valid and log-loss falls -------------
pp = gbc.predict_proba(Xk)
check("class probabilities in [0,1]", all(0.0 <= p <= 1.0 for p in pp))
check("log loss is small after training", gbc.log_loss(Xk, yk) < 0.3)
# more trees reduce log loss (vs a 3-tree model)
gbc_few = GradientBoostingClassifier(n_estimators=3, learning_rate=0.2, max_depth=3).fit(Xk, yk)
check("more trees reduce log loss", gbc.log_loss(Xk, yk) < gbc_few.log_loss(Xk, yk))

# --- base prediction equals the mean (regression) --------------------------
gb0 = GradientBoostingRegressor(n_estimators=0).fit(X, y)
check("zero trees predicts the mean", approx(gb0.predict([[0.0]])[0], sum(y) / len(y), 1e-9))

# --- separable classes: near-perfect -------------------------------------
Xs, ys = [], []
for c, (cx, cy) in enumerate([(0.0, 0.0), (5.0, 5.0)]):
    for _ in range(40):
        Xs.append([cx + (rng() - 0.5) * 2, cy + (rng() - 0.5) * 2])
        ys.append(c)
gbs = GradientBoostingClassifier(n_estimators=40, learning_rate=0.3, max_depth=2).fit(Xs, ys)
check("separable classes near-perfect", gbs.accuracy(Xs, ys) >= 0.98)

# --- determinism -----------------------------------------------------------
check("GBR deterministic",
      GradientBoostingRegressor(n_estimators=10).fit(X, y).predict(X)
      == GradientBoostingRegressor(n_estimators=10).fit(X, y).predict(X))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all gradient_boosting tests passed")
