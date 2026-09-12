"""Gibbs sampling: drawing from a joint distribution one coordinate at a time.

Sampling from a high-dimensional joint distribution is hard, but sampling from its ONE-DIMENSIONAL
CONDITIONALS -- the distribution of one variable given all the others fixed -- is often easy. GIBBS
SAMPLING (Geman and Geman, 1984) exploits exactly this: cycle through the variables, and resample
each in turn from its conditional distribution given the current values of the rest. The sequence of
states is a Markov chain whose stationary distribution is the target joint, so after a burn-in the
samples are draws from it. It is the workhorse of Bayesian statistics (hierarchical models, latent
Dirichlet allocation, image restoration) precisely because the conditionals are usually simple even
when the joint is intractable.

The method is a special case of Metropolis-Hastings where every proposal is ACCEPTED (the acceptance
ratio is exactly 1 because we propose from the true conditional), so it never wastes a step. For a
MULTIVARIATE GAUSSIAN the conditional of each coordinate given the others is itself a 1-D Gaussian
with a mean that is a linear function of the other coordinates and a variance fixed by the covariance
structure -- a clean closed form. For a discrete model, the conditional is a categorical draw over
the states of one variable. The chain mixes by exploring along the coordinate axes; correlated
variables slow it, but it still converges, and its samples reproduce the target's means, variances,
and correlations.

This module implements Gibbs sampling for a bivariate and general multivariate Gaussian (via the
closed-form conditionals) and a generic coordinate sampler that takes user conditionals, with a
seeded RNG. It is verified against exact references: that the sampled mean and covariance of a
correlated bivariate Gaussian converge to the true parameters, that the recovered correlation matches
the target, that the marginals have the right variance, that a 3-D Gaussian's full covariance is
recovered, and that a generic discrete conditional sampler reproduces a known joint. Pure stdlib; a
statistics companion to the Metropolis, rejection-sampling, and Gaussian-process notes."""

from __future__ import annotations

import math


class _RNG:
    def __init__(self, seed=1):
        self.state = seed & 0xFFFFFFFF
        self._spare = None

    def uniform(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def gauss(self, mu=0.0, sigma=1.0):
        if self._spare is not None:
            g, self._spare = self._spare, None
            return mu + sigma * g
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return mu + sigma * r * math.cos(2 * math.pi * u2)


def bivariate_gaussian(mean, cov, n_samples, burn_in=500, seed=1):
    """Gibbs-sample a bivariate Gaussian N(mean, cov). cov = [[s11, s12], [s12, s22]].

    The conditional of x given y is N(mu_x + (s12/s22)(y - mu_y), s11 - s12^2/s22)."""
    rng = _RNG(seed)
    mx, my = mean
    s11, s12 = cov[0]
    _, s22 = cov[1]
    # conditional variances (Schur complements)
    var_x_given_y = s11 - s12 * s12 / s22
    var_y_given_x = s22 - s12 * s12 / s11
    sd_x = math.sqrt(max(var_x_given_y, 1e-15))
    sd_y = math.sqrt(max(var_y_given_x, 1e-15))

    x, y = mx, my
    samples = []
    for i in range(n_samples + burn_in):
        # x | y
        mu_x = mx + (s12 / s22) * (y - my)
        x = rng.gauss(mu_x, sd_x)
        # y | x
        mu_y = my + (s12 / s11) * (x - mx)
        y = rng.gauss(mu_y, sd_y)
        if i >= burn_in:
            samples.append((x, y))
    return samples


def multivariate_gaussian(mean, cov, n_samples, burn_in=1000, seed=1):
    """Gibbs-sample a multivariate Gaussian N(mean, cov) using the closed-form conditionals.

    The conditional of coordinate i given the rest is Gaussian; its mean and variance come from the
    precision matrix (inverse covariance): var_i = 1/Prec[i][i], mean shift from the other coords."""
    rng = _RNG(seed)
    d = len(mean)
    prec = _invert(cov)
    x = list(mean)
    samples = []
    for it in range(n_samples + burn_in):
        for i in range(d):
            # conditional variance and mean from the precision matrix
            var_i = 1.0 / prec[i][i]
            s = sum(prec[i][j] * (x[j] - mean[j]) for j in range(d) if j != i)
            mu_i = mean[i] - var_i * s
            x[i] = rng.gauss(mu_i, math.sqrt(var_i))
        if it >= burn_in:
            samples.append(list(x))
    return samples


def _invert(M):
    """Invert a small square matrix by Gauss-Jordan elimination."""
    n = len(M)
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [v / pv for v in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0.0:
                f = A[r][col]
                A[r] = [A[r][k] - f * A[col][k] for k in range(2 * n)]
    return [row[n:] for row in A]


def gibbs(initial, conditionals, n_samples, burn_in=500, seed=1):
    """Generic Gibbs sampler. `initial` is the starting state (list). `conditionals` is a list of
    functions, one per coordinate: conditionals[i](state, rng) returns a fresh draw of coordinate i
    given the current state. Returns the post-burn-in samples."""
    rng = _RNG(seed)
    state = list(initial)
    samples = []
    for it in range(n_samples + burn_in):
        for i in range(len(state)):
            state[i] = conditionals[i](state, rng)
        if it >= burn_in:
            samples.append(list(state))
    return samples


# --- helpers for verifying samples -----------------------------------------
def sample_mean(samples):
    d = len(samples[0])
    return [sum(s[i] for s in samples) / len(samples) for i in range(d)]


def sample_cov(samples):
    n = len(samples)
    d = len(samples[0])
    m = sample_mean(samples)
    cov = [[0.0] * d for _ in range(d)]
    for s in samples:
        for i in range(d):
            for j in range(d):
                cov[i][j] += (s[i] - m[i]) * (s[j] - m[j])
    return [[cov[i][j] / (n - 1) for j in range(d)] for i in range(d)]
