"""Tests for gmm: parameter recovery, monotone EM, log-sum-exp, model selection."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gmm import GaussianMixture, fit_best, _logsumexp, _log_gauss_diag

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- log-sum-exp numerics --------------------------------------------------
check("logsumexp equals plain log-sum-exp", approx(_logsumexp([0.0, 0.0]), math.log(2.0), 1e-12))
check("logsumexp handles large values", approx(_logsumexp([1000.0, 1000.0]), 1000.0 + math.log(2.0), 1e-9))
check("logsumexp with -inf", approx(_logsumexp([float("-inf"), 0.0]), 0.0, 1e-12))

# --- diagonal Gaussian log-density integrates like a normal ----------------
# at the mean, 1-D N(0,1): log density = -0.5*log(2pi)
check("log gauss at mean", approx(_log_gauss_diag([0.0], [0.0], [1.0]), -0.5 * math.log(2 * math.pi), 1e-12))
# symmetric
check("log gauss symmetric", approx(_log_gauss_diag([1.0], [0.0], [1.0]),
                                    _log_gauss_diag([-1.0], [0.0], [1.0]), 1e-12))

# --- data generator (LCG + Box-Muller, high bits) --------------------------
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def gauss(mu, sd):
    u1 = max(1e-9, rng())
    u2 = rng()
    return mu + sd * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


# two well-separated 2-D clusters with UNEQUAL spread
X = []
for _ in range(150):
    X.append([gauss(0.0, 0.5), gauss(0.0, 0.5)])
for _ in range(150):
    X.append([gauss(6.0, 1.5), gauss(6.0, 1.5)])

g = fit_best(X, 2, restarts=6, seed=1)

# --- recovers the known means (up to label swap) ---------------------------
found = sorted(tuple(m) for m in g.means)
near_origin = found[0]
near_six = found[1]
check("recovered cluster near origin", abs(near_origin[0]) < 0.5 and abs(near_origin[1]) < 0.5)
check("recovered cluster near (6,6)", abs(near_six[0] - 6) < 0.6 and abs(near_six[1] - 6) < 0.6)
check("recovered equal mixing weights", approx(g.weights[0], 0.5, 0.08))

# --- captures the UNEQUAL variances (the thing k-means can't) --------------
# component near origin should have small variance, the (6,6) one large
var_by_mean = sorted(zip([m[0] for m in g.means], g.variances))
tight_var = var_by_mean[0][1]
wide_var = var_by_mean[1][1]
check("tight cluster small variance", tight_var[0] < 1.0)
check("wide cluster large variance", wide_var[0] > 1.0)

# --- EM increases log-likelihood every iteration ---------------------------
mono = all(g.history_[i + 1] >= g.history_[i] - 1e-6 for i in range(len(g.history_) - 1))
check("log-likelihood monotone non-decreasing", mono)
check("converged before max_iter", g.n_iter_ < 200)

# --- responsibilities are valid probabilities ------------------------------
proba = g.predict_proba(X)
check("responsibilities sum to 1", all(approx(sum(row), 1.0, 1e-9) for row in proba))
check("responsibilities in [0,1]", all(0.0 <= v <= 1.0 for row in proba for v in row))

# --- hard labels separate the two clusters ---------------------------------
labels = g.predict(X)
first_half = labels[:150]
second_half = labels[150:]
# each true cluster maps to a single dominant label
check("cluster 1 mostly one label", max(first_half.count(0), first_half.count(1)) >= 145)
check("cluster 2 mostly one label", max(second_half.count(0), second_half.count(1)) >= 145)
check("two clusters get different labels",
      max(set(first_half), key=first_half.count) != max(set(second_half), key=second_half.count))

# --- BIC selects the true number of components (k=2) -----------------------
bics = {}
for k in (1, 2, 3, 4):
    bics[k] = fit_best(X, k, restarts=4, seed=2).bic(X)
check("BIC minimized at true k=2", min(bics, key=bics.get) == 2)
check("k=2 beats k=1 (real structure)", bics[2] < bics[1])

# --- more components fit training data at least as well (higher logL) ------
ll1 = fit_best(X, 1, restarts=3, seed=5).score(X)
ll3 = fit_best(X, 3, restarts=3, seed=5).score(X)
check("more components => higher log-likelihood", ll3 >= ll1 - 1.0)

# --- score equals last fitted log-likelihood -------------------------------
check("score matches fitted log-likelihood", approx(g.score(X), g.log_likelihood_, 1e-6))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all gmm tests passed")
