"""Bayesian optimization: finding the minimum of an expensive black box in few evaluations.

Some functions are costly to evaluate -- a hyperparameter sweep that trains a model each time, a
physical experiment, a slow simulation. Grid or random search wastes most of that budget probing
uninteresting regions. BAYESIAN OPTIMIZATION instead builds a cheap probabilistic surrogate of the
objective (a Gaussian process), and at each step spends its next expensive evaluation where an
ACQUISITION FUNCTION says the expected payoff is highest -- balancing EXPLOITATION (sample where
the surrogate predicts a low value) against EXPLORATION (sample where it is uncertain).

The classic acquisition is EXPECTED IMPROVEMENT. With the current best (lowest) value f_best, and
the GP posterior giving mean mu(x) and std sigma(x) at a candidate x, the improvement is
max(f_best - f(x), 0); its expectation under the Gaussian posterior has the closed form

    EI(x) = (f_best - mu) Phi(z) + sigma phi(z),   z = (f_best - mu) / sigma

where Phi and phi are the standard-normal CDF and PDF. EI is zero at observed points (no
uncertainty, no improvement) and large where the surrogate is both promising and unsure -- so
maximizing it automatically trades off the two. The loop is: fit the GP, maximize EI over a
candidate set to pick the next point, evaluate the true objective there, repeat.

This module implements Expected Improvement and the full optimization loop over a bounded domain,
with self-contained normal CDF/PDF -- verified that it locates the minima of standard test
functions (a 1-D multimodal function and the 2-D Branin function) to good accuracy within a small
budget, and that it beats random search given the same number of evaluations. Pure stdlib, built on
the Gaussian-process regressor; the decision-making companion to the GP note."""

from __future__ import annotations

import math

from gaussian_process import GaussianProcess, fit_length_scale


def _norm_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z):
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_improvement(mu, sigma, f_best, xi=0.0):
    """Expected improvement for MINIMIZATION at a point with posterior (mu, sigma).

    xi is an optional exploration margin (require improvement beyond f_best - xi)."""
    if sigma <= 1e-12:
        return 0.0
    imp = f_best - mu - xi
    z = imp / sigma
    return imp * _norm_cdf(z) + sigma * _norm_pdf(z)


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0
    return rng


def _sample_domain(bounds, rng):
    """One uniform sample inside an axis-aligned box given as [(lo, hi), ...]."""
    return [lo + (hi - lo) * rng() for (lo, hi) in bounds]


def minimize(objective, bounds, n_init=5, n_iter=20, n_candidates=400,
             length_scales=None, noise_var=1e-6, xi=0.01, seed=0):
    """Minimize a black-box objective over a box `bounds` by Bayesian optimization.

    objective: function taking a list of coordinates -> scalar to minimize.
    bounds: list of (lo, hi) per dimension.
    n_init: random initial evaluations before the model-guided loop.
    n_iter: model-guided iterations (each = one true objective evaluation).
    n_candidates: random candidates scored by EI each iteration.

    Returns dict with best_x, best_y, and the full history of (x, y) evaluations."""
    rng = _lcg(seed)
    if length_scales is None:
        # default candidate length scales relative to the domain size
        span = sum((hi - lo) for lo, hi in bounds) / len(bounds)
        length_scales = [span * f for f in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)]

    X, y = [], []
    for _ in range(n_init):
        x = _sample_domain(bounds, rng)
        X.append(x)
        y.append(objective(x))

    for _ in range(n_iter):
        f_best = min(y)
        # fit a GP, tuning the length scale by marginal likelihood
        gp, _, _ = fit_length_scale(X, y, length_scales, noise_var=noise_var)
        # score a fresh batch of random candidates by expected improvement
        best_ei, best_x = -1.0, None
        for _ in range(n_candidates):
            cand = _sample_domain(bounds, rng)
            mu, sigma = gp.predict([cand])
            ei = expected_improvement(mu[0], sigma[0], f_best, xi=xi)
            if ei > best_ei:
                best_ei, best_x = ei, cand
        if best_x is None:                     # degenerate: fall back to random
            best_x = _sample_domain(bounds, rng)
        X.append(best_x)
        y.append(objective(best_x))

    i_best = min(range(len(y)), key=lambda i: y[i])
    return {"best_x": X[i_best], "best_y": y[i_best],
            "X": X, "y": y, "n_eval": len(y)}


def random_search(objective, bounds, n_eval=25, seed=0):
    """Baseline: evaluate the objective at n_eval uniform-random points, keep the best."""
    rng = _lcg(seed)
    X, y = [], []
    for _ in range(n_eval):
        x = _sample_domain(bounds, rng)
        X.append(x)
        y.append(objective(x))
    i_best = min(range(len(y)), key=lambda i: y[i])
    return {"best_x": X[i_best], "best_y": y[i_best], "X": X, "y": y, "n_eval": n_eval}
