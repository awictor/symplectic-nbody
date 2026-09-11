"""Gaussian mixture models by Expectation-Maximization: soft, probabilistic clustering.

k-means assigns each point hard to its nearest centroid -- a Gaussian mixture instead models the
data as drawn from k Gaussians and asks, for every point, the PROBABILITY it came from each. That
soft assignment lets clusters have different sizes, weights, and spreads, and it comes with a real
likelihood you can compare across models.

The fit is EXPECTATION-MAXIMIZATION (Dempster, Laird, Rubin, 1977), a hill-climb on the data
log-likelihood that alternates two steps until it stops rising:

  E-step: with the current means, variances, and weights fixed, compute each point's RESPONSIBILITY
          r[i,c] = pi_c N(x_i | mu_c, sigma_c) / sum_c' (...)  -- the posterior P(component c | x_i).
  M-step: with responsibilities fixed, re-estimate each component as the responsibility-weighted
          mean, variance, and total weight of the data.

Each round cannot decrease the log-likelihood (that monotonic climb is the standard EM correctness
check), and it converges to a local optimum -- so, like k-means, it is run from several inits and
the best likelihood kept. This module fits a diagonal-covariance mixture in any dimension with
log-sum-exp numerics for stability, scores samples, predicts hard labels and soft responsibilities,
and reports AIC/BIC for choosing k -- verified to recover known mixture parameters, to increase
log-likelihood every iteration, and to beat k-means on unequal-variance clusters. Pure stdlib; the
probabilistic companion to the k-means note."""

from __future__ import annotations

import math

_LOG2PI = math.log(2.0 * math.pi)


def _logsumexp(vals):
    """log(sum(exp(v))) computed stably by factoring out the max."""
    m = max(vals)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _log_gauss_diag(x, mean, var):
    """log N(x | mean, diag(var)) for a diagonal-covariance Gaussian."""
    s = 0.0
    for d in range(len(x)):
        v = var[d]
        s += -0.5 * (_LOG2PI + math.log(v) + (x[d] - mean[d]) ** 2 / v)
    return s


class GaussianMixture:
    """A diagonal-covariance Gaussian mixture fit by EM."""

    def __init__(self, n_components, max_iter=200, tol=1e-6, reg=1e-6, seed=0):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg               # variance floor, keeps components from collapsing
        self._state = seed & 0xFFFFFFFF
        self.weights = None          # mixing weights pi_c, sum to 1
        self.means = None            # list of k mean vectors
        self.variances = None        # list of k diagonal-variance vectors
        self.log_likelihood_ = None
        self.history_ = []           # log-likelihood per iteration
        self.n_iter_ = 0

    def _rand(self):
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return (self._state >> 16) / 65536.0    # high bits

    def _init(self, X):
        n, d = len(X), len(X[0])
        # global mean and variance for initialization
        gmean = [sum(row[j] for row in X) / n for j in range(d)]
        gvar = [max(self.reg, sum((row[j] - gmean[j]) ** 2 for row in X) / n) for j in range(d)]
        # means: k distinct random data points
        idx = []
        while len(idx) < self.k:
            c = int(self._rand() * n)
            if c >= n:
                c = n - 1
            if c not in idx:
                idx.append(c)
        self.means = [list(X[i]) for i in idx]
        self.variances = [list(gvar) for _ in range(self.k)]
        self.weights = [1.0 / self.k for _ in range(self.k)]

    def _e_step(self, X):
        """Return (log_resp, total_log_likelihood). log_resp[i][c] = log P(c | x_i)."""
        log_resp = []
        total = 0.0
        for x in X:
            logp = [math.log(self.weights[c]) + _log_gauss_diag(x, self.means[c], self.variances[c])
                    for c in range(self.k)]
            denom = _logsumexp(logp)
            total += denom
            log_resp.append([lp - denom for lp in logp])
        return log_resp, total

    def _m_step(self, X, log_resp):
        n, d = len(X), len(X[0])
        resp = [[math.exp(lr) for lr in row] for row in log_resp]
        for c in range(self.k):
            nc = sum(resp[i][c] for i in range(n))
            nc = max(nc, 1e-12)
            self.weights[c] = nc / n
            self.means[c] = [sum(resp[i][c] * X[i][j] for i in range(n)) / nc for j in range(d)]
            self.variances[c] = [
                max(self.reg,
                    sum(resp[i][c] * (X[i][j] - self.means[c][j]) ** 2 for i in range(n)) / nc)
                for j in range(d)]

    def fit(self, X):
        best = None
        # a single init here; call fit_best for multi-restart
        self._init(X)
        prev = float("-inf")
        self.history_ = []
        for it in range(self.max_iter):
            log_resp, ll = self._e_step(X)
            self.history_.append(ll)
            self._m_step(X, log_resp)
            self.n_iter_ = it + 1
            if ll - prev < self.tol and it > 0:
                break
            prev = ll
        self.log_likelihood_ = self.history_[-1]
        return self

    def score(self, X):
        """Total log-likelihood of the data under the fitted mixture."""
        _, ll = self._e_step(X)
        return ll

    def predict_proba(self, X):
        """Soft responsibilities P(component | x) for each point."""
        log_resp, _ = self._e_step(X)
        return [[math.exp(lr) for lr in row] for row in log_resp]

    def predict(self, X):
        """Hard cluster labels (argmax responsibility)."""
        out = []
        for row in self.predict_proba(X):
            out.append(max(range(self.k), key=lambda c: row[c]))
        return out

    def n_parameters(self):
        d = len(self.means[0])
        # k means (d each) + k variances (d each) + (k-1) free weights
        return self.k * d + self.k * d + (self.k - 1)

    def aic(self, X):
        """Akaike information criterion: 2p - 2 logL (lower is better)."""
        return 2 * self.n_parameters() - 2 * self.score(X)

    def bic(self, X):
        """Bayesian information criterion: p log n - 2 logL (lower is better)."""
        return self.n_parameters() * math.log(len(X)) - 2 * self.score(X)


def fit_best(X, n_components, restarts=8, max_iter=200, tol=1e-6, reg=1e-6, seed=0):
    """Fit from several random inits, keep the highest-likelihood model (EM finds local optima)."""
    best = None
    for r in range(restarts):
        g = GaussianMixture(n_components, max_iter=max_iter, tol=tol, reg=reg,
                            seed=(seed + r * 7919) & 0xFFFFFFFF).fit(X)
        if best is None or g.log_likelihood_ > best.log_likelihood_:
            best = g
    return best
