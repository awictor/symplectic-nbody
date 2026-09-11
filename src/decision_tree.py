"""Decision trees (CART): classification by asking the best yes/no questions.

A decision tree classifies by asking a sequence of threshold questions -- "is feature 3 <= 2.5?"
-- funnelling each sample down to a leaf that predicts a class. It is the most INTERPRETABLE
model: the path from root to leaf is a readable rule, and the tree needs no feature scaling. CART
(Breiman et al., 1984) builds it greedily: at each node, try every feature and every split value,
and keep the split that most reduces the IMPURITY of the resulting children.

Impurity measures how mixed a node's labels are. GINI impurity is 1 - sum p_c^2 (the chance two
random draws differ); ENTROPY is -sum p_c log2 p_c (bits of surprise). Both are 0 for a pure node
and maximal for a uniform mix. The best split maximizes the weighted impurity drop (the
"information gain" for entropy). Recursing until nodes are pure -- or a max depth / minimum node
size stops it -- grows the tree; those stopping rules and later pruning fight the overfitting a
fully grown tree is prone to.

Trees are the building block of random forests and gradient boosting, the workhorses of tabular
machine learning. This module builds a CART classifier with Gini or entropy, predicts, exposes
the learned rules as text, and reports accuracy and feature importances, checked on separable and
XOR-like data and against a known split. Pure stdlib; the supervised-learning companion to the
regression and k-means notes."""

from __future__ import annotations

import math


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "prediction", "counts")

    def __init__(self):
        self.feature = None       # split feature index (None at a leaf)
        self.threshold = None     # split threshold: go left if x[feature] <= threshold
        self.left = None
        self.right = None
        self.prediction = None    # majority class (leaves and internal nodes both store it)
        self.counts = None        # class counts at this node


def gini(labels) -> float:
    """Gini impurity 1 - sum p_c^2 of a label list; 0 if pure."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def entropy(labels) -> float:
    """Shannon entropy -sum p_c log2 p_c of a label list; 0 if pure."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _majority(labels):
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return max(counts, key=lambda k: (counts[k], k)), counts


def _best_split(X, y, impurity, features=None):
    """Find the (feature, threshold) that most reduces impurity. Returns (feature, threshold,
    gain) or (None, None, 0) if no split helps. If `features` is given, only those feature
    indices are considered (random forests sample a subset per node)."""
    n = len(y)
    parent = impurity(y)
    best = (None, None, 0.0)
    d = len(X[0])
    for f in (range(d) if features is None else features):
        # candidate thresholds: midpoints between sorted unique feature values
        vals = sorted(set(row[f] for row in X))
        for i in range(len(vals) - 1):
            thr = (vals[i] + vals[i + 1]) / 2
            left_y = [y[k] for k in range(n) if X[k][f] <= thr]
            right_y = [y[k] for k in range(n) if X[k][f] > thr]
            if not left_y or not right_y:
                continue
            w = (len(left_y) * impurity(left_y) + len(right_y) * impurity(right_y)) / n
            gain = parent - w
            if gain > best[2]:
                best = (f, thr, gain)
    return best


class DecisionTree:
    """A CART decision-tree classifier."""

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 max_features=None, seed=0):
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # max_features: consider only a random subset of features per split (random-forest
        # style). None = all features; "sqrt" = round(sqrt(d)); an int = that many.
        self.max_features = max_features
        self._state = seed & 0xFFFFFFFF
        self.root = None
        self.n_features = 0
        self._importance = None

    def _rand(self):
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return (self._state >> 16) / 65536.0    # high bits

    def _sample_features(self):
        d = self.n_features
        if self.max_features is None:
            return None
        if self.max_features == "sqrt":
            k = max(1, round(d ** 0.5))
        else:
            k = max(1, min(int(self.max_features), d))
        # partial Fisher-Yates on an index list, take first k
        idx = list(range(d))
        for i in range(k):
            j = i + int(self._rand() * (d - i))
            if j >= d:
                j = d - 1
            idx[i], idx[j] = idx[j], idx[i]
        return idx[:k]

    def fit(self, X, y):
        self.n_features = len(X[0])
        self._importance = [0.0] * self.n_features
        self.root = self._build(X, list(y), depth=0)
        # normalize feature importances
        total = sum(self._importance)
        if total > 0:
            self._importance = [v / total for v in self._importance]
        return self

    def _build(self, X, y, depth):
        node = _Node()
        node.prediction, node.counts = _majority(y)
        # stop: pure node, depth cap, or too few samples
        if (len(set(y)) == 1 or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        f, thr, gain = _best_split(X, y, self.criterion, self._sample_features())
        if f is None or gain <= 0:
            return node
        node.feature, node.threshold = f, thr
        self._importance[f] += gain * len(y)      # weight importance by node size
        left_idx = [k for k in range(len(y)) if X[k][f] <= thr]
        right_idx = [k for k in range(len(y)) if X[k][f] > thr]
        node.left = self._build([X[k] for k in left_idx], [y[k] for k in left_idx], depth + 1)
        node.right = self._build([X[k] for k in right_idx], [y[k] for k in right_idx], depth + 1)
        return node

    def _predict_one(self, x):
        node = self.root
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def accuracy(self, X, y):
        preds = self.predict(X)
        return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)

    def depth(self):
        def d(node):
            if node is None or node.feature is None:
                return 0
            return 1 + max(d(node.left), d(node.right))
        return d(self.root)

    def n_leaves(self):
        def count(node):
            if node.feature is None:
                return 1
            return count(node.left) + count(node.right)
        return count(self.root)

    def feature_importances(self):
        """Normalized reduction in impurity attributable to each feature."""
        return list(self._importance)

    def rules(self):
        """The tree as human-readable if/else rules (a list of indented text lines)."""
        lines = []

        def walk(node, depth, prefix):
            indent = "  " * depth
            if node.feature is None:
                lines.append(f"{indent}{prefix}predict {node.prediction}  {dict(node.counts)}")
            else:
                lines.append(f"{indent}{prefix}if x[{node.feature}] <= {node.threshold:.3g}:")
                walk(node.left, depth + 1, "")
                lines.append(f"{indent}else:")
                walk(node.right, depth + 1, "")

        walk(self.root, 0, "")
        return lines
