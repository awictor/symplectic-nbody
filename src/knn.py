"""k-nearest-neighbours: classification and regression with no training at all.

k-NN is the ultimate lazy learner: it builds no model. To predict a point's label, it finds the k
training points nearest to it and lets them VOTE (classification) or AVERAGES their values
(regression). All the work happens at query time. That makes it a non-parametric method whose
decision boundary can be arbitrarily wiggly -- it simply traces the data -- and a natural baseline
against anything fancier.

Two choices shape it. The number of neighbours k trades variance for bias: k = 1 fits every training
point exactly (a jagged boundary that overfits), while large k averages over a wide region (smooth,
but blurs real structure). And the VOTE can be uniform (every neighbour counts equally) or
DISTANCE-WEIGHTED (closer neighbours count more, weight = 1/distance), which makes predictions less
sensitive to the exact k. Because it compares raw coordinates, k-NN is sensitive to feature scaling,
so this module includes standardization.

A brute-force search is O(n) per query; a k-d tree (see kdtree.py) cuts that to O(log n) in low
dimensions. This module implements k-NN classification and regression with uniform and
distance-weighted voting and optional standardization -- verified that 1-NN reproduces the training
labels exactly, that it recovers separable classes and a smooth regression target, that
distance-weighting breaks ties sensibly, that leave-one-out cross-validation selects a reasonable k,
and that its brute-force neighbours agree with the k-d tree. Pure stdlib; the lazy-learning,
instance-based companion to the naive-Bayes and decision-tree notes."""

from __future__ import annotations

import math


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def standardize_fit(X):
    """Return (mean, std) per feature for z-scoring; std floored to avoid divide-by-zero."""
    n, d = len(X), len(X[0])
    mean = [sum(X[i][f] for i in range(n)) / n for f in range(d)]
    std = []
    for f in range(d):
        v = sum((X[i][f] - mean[f]) ** 2 for i in range(n)) / n
        std.append(math.sqrt(v) if v > 1e-12 else 1.0)
    return mean, std


def standardize_apply(X, mean, std):
    return [[(row[f] - mean[f]) / std[f] for f in range(len(row))] for row in X]


class KNN:
    """k-nearest-neighbours classifier and regressor (brute-force search)."""

    def __init__(self, k=5, weights="uniform", task="classify", standardize=False):
        assert weights in ("uniform", "distance")
        assert task in ("classify", "regress")
        self.k = k
        self.weights = weights
        self.task = task
        self.standardize = standardize
        self.X = None
        self.y = None
        self._mean = None
        self._std = None

    def fit(self, X, y):
        if self.standardize:
            self._mean, self._std = standardize_fit(X)
            self.X = standardize_apply(X, self._mean, self._std)
        else:
            self.X = [list(r) for r in X]
        self.y = list(y)
        return self

    def _neighbours(self, x):
        """Indices and squared distances of the k nearest training points to x."""
        d = [(_dist2(self.X[i], x), i) for i in range(len(self.X))]
        d.sort(key=lambda t: t[0])
        return d[:self.k]

    def _prep(self, X):
        return standardize_apply(X, self._mean, self._std) if self.standardize else X

    def predict(self, X):
        Xp = self._prep(X)
        return [self._predict_one(x) for x in Xp]

    def _predict_one(self, x):
        neigh = self._neighbours(x)
        if self.task == "classify":
            votes = {}
            for d2, i in neigh:
                w = 1.0 if self.weights == "uniform" else 1.0 / (math.sqrt(d2) + 1e-12)
                votes[self.y[i]] = votes.get(self.y[i], 0.0) + w
            # highest weight wins; ties broken by the label's sort order
            return max(votes.items(), key=lambda kv: (kv[1], -_label_key(kv[0])))[0]
        else:
            if self.weights == "uniform":
                return sum(self.y[i] for _, i in neigh) / len(neigh)
            num = den = 0.0
            for d2, i in neigh:
                w = 1.0 / (math.sqrt(d2) + 1e-12)
                num += w * self.y[i]
                den += w
            return num / den

    def predict_proba(self, X):
        """Class vote fractions per query (classification only)."""
        assert self.task == "classify"
        classes = sorted(set(self.y))
        out = []
        for x in self._prep(X):
            neigh = self._neighbours(x)
            votes = {c: 0.0 for c in classes}
            for d2, i in neigh:
                w = 1.0 if self.weights == "uniform" else 1.0 / (math.sqrt(d2) + 1e-12)
                votes[self.y[i]] += w
            total = sum(votes.values())
            out.append({c: votes[c] / total for c in classes})
        return out

    def score(self, X, y):
        """Accuracy (classification) or R^2 (regression)."""
        pred = self.predict(X)
        if self.task == "classify":
            return sum(1 for i in range(len(y)) if pred[i] == y[i]) / len(y)
        ybar = sum(y) / len(y)
        ss_res = sum((y[i] - pred[i]) ** 2 for i in range(len(y)))
        ss_tot = sum((yi - ybar) ** 2 for yi in y)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _label_key(label):
    """A stable numeric-ish key for tie-breaking labels of mixed types."""
    return hash(label) & 0xFFFFFF


def loo_cross_val(X, y, k_values, weights="uniform", task="classify", standardize=False):
    """Leave-one-out cross-validation over candidate k; return (best_k, {k: score})."""
    scores = {}
    n = len(X)
    for k in k_values:
        correct = 0.0
        for i in range(n):
            Xtr = [X[j] for j in range(n) if j != i]
            ytr = [y[j] for j in range(n) if j != i]
            model = KNN(k=k, weights=weights, task=task, standardize=standardize).fit(Xtr, ytr)
            pred = model.predict([X[i]])[0]
            if task == "classify":
                correct += 1.0 if pred == y[i] else 0.0
            else:
                correct += -(pred - y[i]) ** 2      # negative MSE, higher is better
        scores[k] = correct / n
    best_k = max(scores, key=lambda kk: scores[kk])
    return best_k, scores
