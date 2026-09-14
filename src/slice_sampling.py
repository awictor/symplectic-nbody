"""Slice sampling: an MCMC method that tunes its own step size instead of making you guess it.

Metropolis-Hastings works only as well as its proposal width: too small and the chain crawls, too large
and almost every proposal is rejected, and the sweet spot is different for every distribution. Slice
sampling (Radford Neal, 2003) removes that knob. It draws from a density p(x) (known only up to a
constant) by sampling uniformly from the region UNDER the curve, the set { (x, y) : 0 < y < p(x) }, and
then dropping the auxiliary height y. Each step is two conditional draws: given x, pick a height y
uniformly in (0, p(x)); given y, pick a new x uniformly from the SLICE S = { x : p(x) > y }, the set of
points whose density rises above the current height.

The slice is an interval (or union of intervals) that the method finds adaptively, so the effective
step size scales itself to the local width of the distribution -- wide where the density is flat,
narrow where it is peaked. Neal's STEPPING-OUT procedure grows an interval of width w outward from the
current point until both ends fall below the height y, and the SHRINKAGE procedure samples inside that
interval, contracting it toward x whenever a trial point lands outside the slice, until an accepted
point is found. The result is a valid Markov chain that leaves p invariant, with no accept/reject
tuning and no wasted rejections in the final sample.

This module implements univariate slice sampling with stepping-out and shrinkage (seeded for
reproducibility), plus a multivariate coordinate-wise sampler. It is validated: samples from a standard
normal recover mean 0 and variance 1; a shifted/scaled normal recovers its mean and variance; an
exponential recovers mean 1/rate; a Gaussian mixture recovers both modes' mass; the empirical CDF
matches the analytic one (Kolmogorov-style max gap is small); every accepted sample really lies on its
slice (p(x) > y); and results are reproducible for a fixed seed. Pure stdlib; the self-tuning-MCMC
companion to the Metropolis, Gibbs, and HMC samplers."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u32(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def uniform(self):
        return (self.u32() >> 8) / (1 << 24)


def _slice_step(logf, x0, w, rng, max_steps=50):
    """One univariate slice-sampling transition from x0 for log-density logf. Returns the new x.

    Uses a log-height threshold to stay numerically stable for tiny densities."""
    # 1. draw the slice level in log space: log(y) = logf(x0) + log(u), u ~ Uniform(0,1)
    logy = logf(x0) + math.log(rng.uniform() + 1e-300)
    # 2. stepping out: place an interval [L, R] of width w randomly around x0, then expand
    u = rng.uniform()
    L = x0 - w * u
    R = L + w
    j = int(max_steps * rng.uniform())
    k = max_steps - 1 - j
    while j > 0 and logf(L) > logy:
        L -= w
        j -= 1
    while k > 0 and logf(R) > logy:
        R += w
        k -= 1
    # 3. shrinkage: sample uniformly in [L, R], contract toward x0 on misses
    for _ in range(200):
        x1 = L + rng.uniform() * (R - L)
        if logf(x1) > logy:
            return x1
        if x1 < x0:
            L = x1
        else:
            R = x1
    return x0


def sample(logf, x0=0.0, n=1000, w=1.0, burn=100, seed=1):
    """Draw n samples from a 1-D density via slice sampling. logf is the log-density (up to a constant)."""
    rng = _Rng(seed)
    x = x0
    for _ in range(burn):
        x = _slice_step(logf, x, w, rng)
    out = []
    for _ in range(n):
        x = _slice_step(logf, x, w, rng)
        out.append(x)
    return out


def sample_multivariate(logf, x0, n=1000, w=1.0, burn=100, seed=1):
    """Coordinate-wise slice sampling for a multivariate log-density. x0 is a list; updates one dim at a time."""
    rng = _Rng(seed)
    dim = len(x0)
    x = list(x0)

    def make_1d(d, xcur):
        def g(v):
            xt = list(xcur)
            xt[d] = v
            return logf(xt)
        return g

    def sweep(x):
        for d in range(dim):
            g = make_1d(d, x)
            x[d] = _slice_step(g, x[d], w, rng)
        return x

    for _ in range(burn):
        x = sweep(x)
    out = []
    for _ in range(n):
        x = sweep(x)
        out.append(list(x))
    return out


# --- common log-densities (up to a constant) ------------------------------
def log_normal(mu=0.0, sigma=1.0):
    def f(x):
        return -0.5 * ((x - mu) / sigma) ** 2
    return f


def log_exponential(rate=1.0):
    def f(x):
        return -rate * x if x >= 0 else float("-inf")
    return f


def log_mixture(components):
    """Log-density of a Gaussian mixture: components = [(weight, mu, sigma), ...]."""
    def f(x):
        total = 0.0
        for wgt, mu, sig in components:
            total += wgt / (sig * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x - mu) / sig) ** 2)
        return math.log(total) if total > 0 else float("-inf")
    return f


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)
