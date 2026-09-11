"""Gradient boosting: turning many shallow trees into one strong predictor.

A random forest averages independent deep trees; GRADIENT BOOSTING (Friedman, 2001) instead grows
trees in SEQUENCE, each one correcting the mistakes of those before it. It is gradient descent in
FUNCTION space: start with a constant prediction, then repeatedly fit a small tree to the negative
gradient of the loss (the direction that most reduces error) and add a shrunken step of it to the
running model. For squared-error regression that negative gradient is simply the RESIDUAL y - F(x),
so each tree literally learns what the current ensemble still gets wrong; for logistic
classification it is y - sigmoid(F(x)), the probability error.

Three knobs control the bias-variance trade. The number of trees adds capacity; the LEARNING RATE
shrinks each tree's contribution (small rates need more trees but generalize better -- shrinkage is
regularization); and the tree DEPTH caps how many feature interactions each weak learner can
capture (depth-1 "stumps" are additive, deeper trees model interactions). Because it optimizes the
loss directly and adds capacity gradually, boosting is the method that wins most tabular-data
competitions.

This module implements gradient boosting for squared-error regression and log-loss binary
classification over self-contained regression trees, with staged predictions to watch the loss
fall -- verified that training loss decreases monotonically as trees are added, that the ensemble
beats a single tree on a noisy nonlinear target, that a smaller learning rate needs more trees, and
that it recovers a clean step/sine function and separates classes. Pure stdlib; the sequential-
ensemble companion to the random-forest and decision-tree notes."""

from __future__ import annotations

import math


# --- a minimal CART regression tree (fits continuous targets by squared error) ---
class _RegNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None       # mean target at a leaf


def _variance_reduction_split(X, g, idx):
    """Best (feature, threshold) that most reduces the sum of squared deviations of g over idx."""
    n = len(idx)
    d = len(X[0])
    total = sum(g[i] for i in idx)
    # parent SSE
    mean = total / n
    parent = sum((g[i] - mean) ** 2 for i in idx)
    best = (None, None, 0.0, None, None)
    for f in range(d):
        order = sorted(idx, key=lambda i: X[i][f])
        # prefix sums to evaluate every threshold in one sweep
        left_sum = 0.0
        left_sq = 0.0
        for s in range(n - 1):
            i = order[s]
            left_sum += g[i]
            left_sq += g[i] * g[i]
            nl = s + 1
            nr = n - nl
            if X[order[s]][f] == X[order[s + 1]][f]:
                continue                      # can't split between equal values
            right_sum = total - left_sum
            # SSE = sum sq - sum^2 / count  for each side
            sse_l = left_sq - left_sum * left_sum / nl
            right_sq = sum(g[order[t]] ** 2 for t in range(nl, n))
            sse_r = right_sq - right_sum * right_sum / nr
            gain = parent - (sse_l + sse_r)
            if gain > best[2]:
                thr = (X[order[s]][f] + X[order[s + 1]][f]) / 2
                best = (f, thr, gain, order[:nl], order[nl:])
    return best


class _RegTree:
    def __init__(self, max_depth=3, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def fit(self, X, g):
        self.root = self._build(X, g, list(range(len(X))), 0)
        return self

    def _build(self, X, g, idx, depth):
        node = _RegNode()
        node.value = sum(g[i] for i in idx) / len(idx)
        if depth >= self.max_depth or len(idx) < self.min_samples or len(set(g[i] for i in idx)) == 1:
            return node
        f, thr, gain, left_idx, right_idx = _variance_reduction_split(X, g, idx)
        if f is None or gain <= 1e-12:
            return node
        node.feature, node.threshold = f, thr
        node.left = self._build(X, g, left_idx, depth + 1)
        node.right = self._build(X, g, right_idx, depth + 1)
        return node

    def predict_one(self, x):
        node = self.root
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return [self.predict_one(x) for x in X]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class GradientBoostingRegressor:
    """Gradient boosting for squared-error regression."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.init_ = 0.0
        self.trees = []

    def fit(self, X, y):
        self.init_ = sum(y) / len(y)              # optimal constant for squared error = mean
        F = [self.init_] * len(y)
        self.trees = []
        for _ in range(self.n_estimators):
            residual = [y[i] - F[i] for i in range(len(y))]   # negative gradient of 0.5(y-F)^2
            tree = _RegTree(self.max_depth, self.min_samples).fit(X, residual)
            step = tree.predict(X)
            for i in range(len(y)):
                F[i] += self.learning_rate * step[i]
            self.trees.append(tree)
        return self

    def predict(self, X):
        out = [self.init_] * len(X)
        for tree in self.trees:
            step = tree.predict(X)
            for i in range(len(X)):
                out[i] += self.learning_rate * step[i]
        return out

    def staged_predict(self, X):
        """Yield the ensemble's prediction after each added tree (to watch the loss fall)."""
        out = [self.init_] * len(X)
        preds = []
        for tree in self.trees:
            step = tree.predict(X)
            for i in range(len(X)):
                out[i] += self.learning_rate * step[i]
            preds.append(list(out))
        return preds

    def mse(self, X, y):
        p = self.predict(X)
        return sum((y[i] - p[i]) ** 2 for i in range(len(y))) / len(y)

    def r_squared(self, X, y):
        p = self.predict(X)
        ybar = sum(y) / len(y)
        ss_res = sum((y[i] - p[i]) ** 2 for i in range(len(y)))
        ss_tot = sum((yi - ybar) ** 2 for yi in y)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class GradientBoostingClassifier:
    """Gradient boosting for binary classification with logistic (log) loss.

    Labels must be 0/1. Trees are fit to the residual y - sigmoid(F), the negative gradient of the
    log loss, and predictions are the sigmoid of the additive score."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.init_ = 0.0
        self.trees = []

    def fit(self, X, y):
        p = sum(y) / len(y)
        p = min(max(p, 1e-6), 1 - 1e-6)
        self.init_ = math.log(p / (1 - p))         # log-odds of the base rate
        F = [self.init_] * len(y)
        self.trees = []
        for _ in range(self.n_estimators):
            residual = [y[i] - _sigmoid(F[i]) for i in range(len(y))]  # neg gradient of log loss
            tree = _RegTree(self.max_depth, self.min_samples).fit(X, residual)
            step = tree.predict(X)
            for i in range(len(y)):
                F[i] += self.learning_rate * step[i]
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        out = [self.init_] * len(X)
        for tree in self.trees:
            step = tree.predict(X)
            for i in range(len(X)):
                out[i] += self.learning_rate * step[i]
        return out

    def predict_proba(self, X):
        return [_sigmoid(z) for z in self.decision_function(X)]

    def predict(self, X):
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    def accuracy(self, X, y):
        pred = self.predict(X)
        return sum(1 for i in range(len(y)) if pred[i] == y[i]) / len(y)

    def log_loss(self, X, y):
        eps = 1e-12
        p = self.predict_proba(X)
        return -sum(y[i] * math.log(p[i] + eps) + (1 - y[i]) * math.log(1 - p[i] + eps)
                    for i in range(len(y))) / len(y)
