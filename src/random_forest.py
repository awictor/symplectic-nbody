"""Random forests: a committee of decorrelated decision trees.

A single decision tree overfits -- it will carve the training data into pure leaves and memorize
noise. A RANDOM FOREST (Breiman, 2001) fixes this by averaging many trees that are each
deliberately weakened and made to disagree, so their errors cancel while their signal adds. Two
sources of randomness decorrelate the trees:

  1. BAGGING (bootstrap aggregating): each tree trains on a bootstrap sample -- n points drawn
     WITH replacement from the n training rows, so ~63% of rows appear (1 - 1/e) and the rest are
     "out-of-bag" for that tree.
  2. FEATURE SUBSAMPLING: at every split a tree considers only a random subset of features
     (classically round(sqrt(d))), so no single strong feature dominates every tree.

Prediction is a majority vote across the trees. Because the ~37% out-of-bag rows of each tree were
never seen by it, averaging each row's vote over only the trees that did NOT train on it gives the
OUT-OF-BAG error -- a free, honest validation estimate needing no held-out set. Feature importance
is the mean of the per-tree impurity-reduction importances.

Forests are the go-to strong baseline for tabular data: little tuning, no scaling, robust to noise
features, and hard to overfit as trees are added. This module builds a bagged forest of the
decision_tree CART learner with per-node feature sampling, majority-vote prediction, out-of-bag
scoring, and averaged feature importances -- verified to beat a single tree on a noisy problem and
to have its OOB estimate track true test error. Pure stdlib; the ensemble companion to the
decision-tree note."""

from __future__ import annotations

from decision_tree import DecisionTree


def _majority_vote(votes):
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: (counts[k], k))


class RandomForest:
    """A bagged random-forest classifier over CART decision trees."""

    def __init__(self, n_trees=25, criterion="gini", max_depth=None,
                 min_samples_split=2, max_features="sqrt", seed=0):
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self._state = seed & 0xFFFFFFFF
        self.trees = []
        self.oob_indices = []      # per tree: the row indices NOT in its bootstrap sample
        self.n_features = 0
        self.oob_score_ = None

    def _rand(self):
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return (self._state >> 16) / 65536.0   # high bits

    def _bootstrap(self, n):
        """n indices drawn with replacement; return (in_bag_indices, out_of_bag_set)."""
        in_bag = [int(self._rand() * n) for _ in range(n)]
        for k in range(len(in_bag)):
            if in_bag[k] >= n:
                in_bag[k] = n - 1
        oob = set(range(n)) - set(in_bag)
        return in_bag, oob

    def fit(self, X, y):
        n = len(X)
        self.n_features = len(X[0])
        self.trees = []
        self.oob_indices = []
        for t in range(self.n_trees):
            in_bag, oob = self._bootstrap(n)
            Xb = [X[i] for i in in_bag]
            yb = [y[i] for i in in_bag]
            # each tree gets its own derived seed so feature sampling differs but is reproducible
            tree = DecisionTree(criterion=self.criterion, max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=self.max_features,
                                seed=(self._state ^ (t * 2654435761)) & 0xFFFFFFFF)
            tree.fit(Xb, yb)
            self.trees.append(tree)
            self.oob_indices.append(oob)
        self._compute_oob(X, y)
        return self

    def predict(self, X):
        # transpose: gather every tree's prediction for each row, then vote
        preds = [tree.predict(X) for tree in self.trees]
        out = []
        for i in range(len(X)):
            out.append(_majority_vote([preds[t][i] for t in range(len(self.trees))]))
        return out

    def accuracy(self, X, y):
        p = self.predict(X)
        return sum(1 for i in range(len(y)) if p[i] == y[i]) / len(y)

    def _compute_oob(self, X, y):
        """Out-of-bag score: each row voted only by trees that did not train on it."""
        n = len(X)
        correct = 0
        counted = 0
        for i in range(n):
            votes = []
            for t, tree in enumerate(self.trees):
                if i in self.oob_indices[t]:
                    votes.append(tree.predict([X[i]])[0])
            if votes:
                counted += 1
                if _majority_vote(votes) == y[i]:
                    correct += 1
        self.oob_score_ = (correct / counted) if counted else None

    def feature_importances(self):
        """Mean of the per-tree normalized impurity-reduction importances."""
        agg = [0.0] * self.n_features
        for tree in self.trees:
            for f, v in enumerate(tree.feature_importances()):
                agg[f] += v
        total = sum(agg)
        if total > 0:
            agg = [v / total for v in agg]
        return agg
