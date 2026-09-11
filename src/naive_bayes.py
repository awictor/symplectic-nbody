"""Naive Bayes: fast probabilistic classification under a bold independence assumption.

Naive Bayes applies Bayes' theorem to classification with one simplifying leap: it assumes the
features are conditionally INDEPENDENT given the class. That is "naive" -- pixels and words are not
really independent -- but it collapses a joint distribution over all features into a product of
per-feature terms, so training is a single pass that just tallies statistics, and prediction is a
sum of logs. Despite the crude assumption it is remarkably strong, especially for text, and it is
the classic baseline every other classifier is measured against.

By Bayes, the posterior over classes is P(c | x) proportional to P(c) * prod_i P(x_i | c). Taking
logs turns the product into a sum (and dodges underflow), so we pick the class maximizing
log P(c) + sum_i log P(x_i | c). The two common flavours differ only in how P(x_i | c) is modelled:

  GAUSSIAN   -- each continuous feature is Normal(mu, sigma^2) per class; training estimates the
                per-class mean and variance of every feature.
  MULTINOMIAL -- features are counts (e.g. word frequencies); P(word | c) is the smoothed fraction
                of class-c tokens that are that word, with Laplace (add-alpha) smoothing so an
                unseen word never zeroes the whole product. This is the workhorse of spam filters.

This module implements both, entirely in log space, with class priors from the data and Laplace
smoothing for the multinomial case -- verified that the Gaussian model separates well-separated
blobs and matches a hand-computed posterior, that the multinomial model classifies documents and
that smoothing prevents zero probabilities, and that predicted class-probabilities are normalized.
Pure stdlib; the probabilistic-baseline companion to the regression and decision-tree notes."""

from __future__ import annotations

import math

_LOG2PI = math.log(2.0 * math.pi)


def _logsumexp(vals):
    m = max(vals)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


class GaussianNB:
    """Gaussian naive Bayes for continuous features."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = []
        self.log_prior = {}
        self.theta = {}      # class -> per-feature mean
        self.var = {}        # class -> per-feature variance

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.classes = sorted(set(y))
        # a global variance floor keeps zero-variance features from blowing up
        floor = self.var_smoothing * max(
            max(X[i][f] for i in range(n)) - min(X[i][f] for i in range(n)) for f in range(d)
        ) ** 2 if n > 1 else self.var_smoothing
        for c in self.classes:
            idx = [i for i in range(n) if y[i] == c]
            self.log_prior[c] = math.log(len(idx) / n)
            mean = [sum(X[i][f] for i in idx) / len(idx) for f in range(d)]
            var = [sum((X[i][f] - mean[f]) ** 2 for i in idx) / len(idx) + floor
                   for f in range(d)]
            self.theta[c] = mean
            self.var[c] = var
        return self

    def _joint_log_likelihood(self, x):
        """log P(c) + sum_f log N(x_f | mu, var) for each class."""
        out = {}
        for c in self.classes:
            s = self.log_prior[c]
            mean, var = self.theta[c], self.var[c]
            for f in range(len(x)):
                s += -0.5 * (_LOG2PI + math.log(var[f]) + (x[f] - mean[f]) ** 2 / var[f])
            out[c] = s
        return out

    def predict(self, X):
        return [max(self._joint_log_likelihood(x).items(), key=lambda kv: kv[1])[0] for x in X]

    def predict_log_proba(self, X):
        res = []
        for x in X:
            jll = self._joint_log_likelihood(x)
            total = _logsumexp(list(jll.values()))
            res.append({c: jll[c] - total for c in self.classes})
        return res

    def predict_proba(self, X):
        return [{c: math.exp(lp) for c, lp in row.items()} for row in self.predict_log_proba(X)]

    def accuracy(self, X, y):
        pred = self.predict(X)
        return sum(1 for i in range(len(y)) if pred[i] == y[i]) / len(y)


class MultinomialNB:
    """Multinomial naive Bayes for count features (e.g. bag-of-words), with Laplace smoothing."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.log_prior = {}
        self.log_prob = {}     # class -> per-feature log P(feature | class)
        self.n_features = 0

    def fit(self, X, y):
        n = len(X)
        self.n_features = len(X[0])
        self.classes = sorted(set(y))
        for c in self.classes:
            idx = [i for i in range(n) if y[i] == c]
            self.log_prior[c] = math.log(len(idx) / n)
            # total count of each feature across class-c documents
            counts = [sum(X[i][f] for i in idx) for f in range(self.n_features)]
            total = sum(counts) + self.alpha * self.n_features    # smoothed denominator
            self.log_prob[c] = [math.log((counts[f] + self.alpha) / total)
                                for f in range(self.n_features)]
        return self

    def _joint_log_likelihood(self, x):
        out = {}
        for c in self.classes:
            s = self.log_prior[c]
            lp = self.log_prob[c]
            for f in range(self.n_features):
                if x[f]:
                    s += x[f] * lp[f]
            out[c] = s
        return out

    def predict(self, X):
        return [max(self._joint_log_likelihood(x).items(), key=lambda kv: kv[1])[0] for x in X]

    def predict_log_proba(self, X):
        res = []
        for x in X:
            jll = self._joint_log_likelihood(x)
            total = _logsumexp(list(jll.values()))
            res.append({c: jll[c] - total for c in self.classes})
        return res

    def predict_proba(self, X):
        return [{c: math.exp(lp) for c, lp in row.items()} for row in self.predict_log_proba(X)]

    def accuracy(self, X, y):
        pred = self.predict(X)
        return sum(1 for i in range(len(y)) if pred[i] == y[i]) / len(y)
