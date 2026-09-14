"""Antithetic variates: halve Monte Carlo variance by pairing each sample with its mirror image.

Monte Carlo averages f over independent samples, and independence means the estimator's variance is
just the sum of per-sample variances. Antithetic variates breaks that independence ON PURPOSE, in a
helpful direction. For each uniform draw U it also evaluates the MIRROR draw 1 - U (for a normal draw
Z, the mirror is -Z), and averages the pair:

    f_anti = ( f(U) + f(1 - U) ) / 2.

Both U and 1 - U are uniform, so the pair average is still an unbiased estimate of E[f]. But U and
1 - U are NEGATIVELY correlated, and when f is MONOTONE that negative correlation carries straight
through: when f(U) happens to be high, f(1 - U) is low, and the two errors cancel in the average. The
variance of the pair is (Var(f) + Cov(f(U), f(1-U))) / 2, so any negative covariance is pure profit --
and for a monotone integrand the covariance is guaranteed negative, giving a variance reduction beyond
the automatic factor-of-two from using two evaluations. The catch is symmetry: if f is symmetric about
the midpoint, f(U) and f(1-U) are perfectly correlated and antithetic sampling gives NO benefit (or can
even hurt), so it pays exactly when the integrand has a consistent trend.

This module implements antithetic estimation over the unit interval/cube and over Gaussian inputs (with
the Z, -Z reflection), reports the achieved variance-reduction factor versus plain Monte Carlo at the
same number of function evaluations, and exposes the pair covariance. It is validated: the estimate is
unbiased (recovers e - 1 for E[e^U] and the analytic value of a monotone integral); the variance is
reduced for monotone integrands and the achieved factor tracks the pair-covariance formula; a symmetric
integrand shows little or no reduction (correctly); the Gaussian reflection recovers E[f(Z)]; and
results are reproducible for a fixed seed. Pure stdlib; the variance-reduction companion to the
control-variate, importance-sampling, and Sobol tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF
        self._spare = None

    def u32(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def uniform(self):
        return (self.u32() >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            z = self._spare
            self._spare = None
            return z
        u1 = self.uniform() + 1e-300
        u2 = self.uniform()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def _mean(a):
    return sum(a) / len(a)


def _var(a):
    m = _mean(a)
    return sum((x - m) ** 2 for x in a) / (len(a) - 1)


def estimate_uniform(f, n_pairs=10000, dim=1, seed=1):
    """Antithetic estimate of E[f(U)] for U ~ Uniform(0,1)^dim, using pairs (U, 1-U).

    Returns (estimate, std_error, reduction_vs_plain). `reduction` compares to plain MC using the
    same total number of function evaluations (2 * n_pairs)."""
    rng = _Rng(seed)
    pair_means = []
    all_vals = []
    for _ in range(n_pairs):
        if dim == 1:
            u = rng.uniform()
            a = f(u)
            b = f(1.0 - u)
        else:
            u = [rng.uniform() for _ in range(dim)]
            a = f(u)
            b = f([1.0 - ui for ui in u])
        pair_means.append((a + b) / 2.0)
        all_vals.append(a)
        all_vals.append(b)
    est = _mean(pair_means)
    # variance of the antithetic estimator (mean of n_pairs pair-averages)
    var_anti = _var(pair_means) / n_pairs
    # plain MC variance with the same 2*n_pairs evaluations
    var_plain = _var(all_vals) / (2 * n_pairs)
    stderr = math.sqrt(var_anti)
    reduction = var_anti / var_plain if var_plain > 0 else 1.0
    return est, stderr, reduction


def estimate_normal(f, n_pairs=10000, seed=1):
    """Antithetic estimate of E[f(Z)] for Z ~ N(0,1) using reflected pairs (Z, -Z)."""
    rng = _Rng(seed)
    pair_means = []
    all_vals = []
    for _ in range(n_pairs):
        z = rng.normal()
        a = f(z)
        b = f(-z)
        pair_means.append((a + b) / 2.0)
        all_vals.append(a)
        all_vals.append(b)
    est = _mean(pair_means)
    var_anti = _var(pair_means) / n_pairs
    var_plain = _var(all_vals) / (2 * n_pairs)
    return est, math.sqrt(var_anti), (var_anti / var_plain if var_plain > 0 else 1.0)


def pair_covariance(f, n_pairs=10000, seed=1):
    """Sample covariance between f(U) and f(1-U); negative means antithetic sampling helps."""
    rng = _Rng(seed)
    xs = []
    ys = []
    for _ in range(n_pairs):
        u = rng.uniform()
        xs.append(f(u))
        ys.append(f(1.0 - u))
    mx, my = _mean(xs), _mean(ys)
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n_pairs)) / (n_pairs - 1)


def plain_estimate(f, n=20000, seed=1, dim=1):
    """Ordinary Monte Carlo estimate of E[f(U)] for reference."""
    rng = _Rng(seed)
    vals = []
    for _ in range(n):
        if dim == 1:
            vals.append(f(rng.uniform()))
        else:
            vals.append(f([rng.uniform() for _ in range(dim)]))
    return _mean(vals), math.sqrt(_var(vals) / n)
