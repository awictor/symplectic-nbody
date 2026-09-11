"""Gaussian process regression: nonparametric prediction with honest uncertainty.

Where linear and polynomial regression fit a FIXED set of coefficients, a Gaussian process fits a
distribution over FUNCTIONS directly, and returns not just a prediction but a calibrated error bar
that widens where there is no data. The idea: assume any finite set of function values is jointly
Gaussian, with covariance given by a KERNEL k(x, x') that says how much two inputs' outputs should
correlate -- nearby points strongly, distant points weakly. The popular RBF (squared-exponential)
kernel k(x,x') = sigma_f^2 exp(-||x-x'||^2 / 2 l^2) encodes "smooth, with length scale l".

Conditioning that joint Gaussian on the observed data gives the POSTERIOR, in closed form:

    mean(x*)  = k*' (K + sigma_n^2 I)^-1 y
    var(x*)   = k(x*,x*) - k*' (K + sigma_n^2 I)^-1 k*

where K is the train-train kernel matrix and k* the train-to-test covariances. The one linear
solve is done via a CHOLESKY factorization of (K + noise), which is symmetric positive definite --
reused for both the mean and the log MARGINAL LIKELIHOOD, whose value scores a choice of
hyperparameters (length scale, signal, noise) so they can be tuned by grid search. With zero noise
the posterior interpolates the training points exactly and its variance there is zero; with noise
it smooths, and everywhere far from data the variance rises back toward the prior.

This module builds GP regression with the RBF kernel, posterior mean and variance, sampling from
the posterior, and the log marginal likelihood -- verified that it interpolates noise-free data
exactly with zero variance there, that uncertainty grows away from the data, that it recovers a
known smooth function, and that the marginal likelihood peaks near the true length scale. Pure
stdlib, built on the Cholesky solver; the Bayesian companion to the regression note."""

from __future__ import annotations

import math

from lu import cholesky, _forward_sub, _back_sub


def rbf_kernel(a, b, length_scale=1.0, signal_var=1.0):
    """Squared-exponential covariance between two input vectors (or scalars)."""
    if not isinstance(a, (list, tuple)):
        a = [a]
    if not isinstance(b, (list, tuple)):
        b = [b]
    sq = sum((a[i] - b[i]) ** 2 for i in range(len(a)))
    return signal_var * math.exp(-sq / (2.0 * length_scale ** 2))


def _kernel_matrix(X1, X2, length_scale, signal_var):
    return [[rbf_kernel(x1, x2, length_scale, signal_var) for x2 in X2] for x1 in X1]


class GaussianProcess:
    """Gaussian process regression with an RBF kernel and Gaussian observation noise."""

    def __init__(self, length_scale=1.0, signal_var=1.0, noise_var=1e-8):
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var        # a tiny jitter also keeps K numerically PD
        self.X = None
        self.y = None
        self.L = None                     # Cholesky factor of (K + noise I)
        self.alpha = None                 # (K + noise I)^-1 y

    def fit(self, X, y):
        self.X = [list(x) if isinstance(x, (list, tuple)) else [x] for x in X]
        self.y = list(y)
        n = len(X)
        K = _kernel_matrix(self.X, self.X, self.length_scale, self.signal_var)
        for i in range(n):
            K[i][i] += self.noise_var
        self.L = cholesky(K)                      # K = L L'  (SPD by construction)
        # alpha = K^-1 y  via two triangular solves
        w = _forward_sub(self.L, self.y)
        Lt = [[self.L[j][i] for j in range(n)] for i in range(n)]
        self.alpha = _back_sub(Lt, w)
        return self

    def predict(self, Xstar, return_std=True):
        """Posterior mean (and standard deviation) at the query points."""
        Xs = [list(x) if isinstance(x, (list, tuple)) else [x] for x in Xstar]
        n = len(self.X)
        Ks = _kernel_matrix(Xs, self.X, self.length_scale, self.signal_var)   # test x train
        means = [sum(Ks[i][j] * self.alpha[j] for j in range(n)) for i in range(len(Xs))]
        if not return_std:
            return means
        stds = []
        Lt = None
        for i in range(len(Xs)):
            # v = L^-1 k*   ->  var = k(x*,x*) - v'v
            v = _forward_sub(self.L, Ks[i])
            kss = self.signal_var                    # rbf_kernel(x*, x*) = signal_var
            var = kss - sum(vj * vj for vj in v)
            stds.append(math.sqrt(max(var, 0.0)))
        return means, stds

    def log_marginal_likelihood(self):
        """log p(y | X, hyperparameters) -- the model-evidence score for hyperparameter tuning."""
        n = len(self.X)
        # -0.5 y' alpha - sum(log L_ii) - n/2 log 2pi
        data_fit = -0.5 * sum(self.y[i] * self.alpha[i] for i in range(n))
        complexity = -sum(math.log(self.L[i][i]) for i in range(n))
        return data_fit + complexity - 0.5 * n * math.log(2.0 * math.pi)


def fit_length_scale(X, y, candidates, signal_var=1.0, noise_var=1e-6):
    """Pick the length scale maximizing the log marginal likelihood over a candidate grid."""
    best = None
    for ls in candidates:
        gp = GaussianProcess(length_scale=ls, signal_var=signal_var, noise_var=noise_var).fit(X, y)
        lml = gp.log_marginal_likelihood()
        if best is None or lml > best[1]:
            best = (ls, lml, gp)
    return best[2], best[0], best[1]
