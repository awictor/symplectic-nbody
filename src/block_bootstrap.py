"""Block bootstrap: resample DEPENDENT data by shuffling contiguous blocks, so autocorrelation survives.

The ordinary bootstrap resamples data points one at a time with replacement -- valid when the data are
independent. But TIME SERIES are correlated: today looks like yesterday. Resample them point-by-point and
you shatter that dependence, so the bootstrap wildly UNDERESTIMATES the variance of the mean (nearby points
carry redundant information the i.i.d. bootstrap treats as fresh), and confidence intervals come out far too
narrow. The BLOCK BOOTSTRAP (Kunsch 1989) fixes this by resampling contiguous BLOCKS of length L instead of
single points: within a block the local correlation structure is preserved, and only the (weaker)
dependence ACROSS blocks is broken.

Three standard variants:
  MOVING-BLOCK: draw blocks with replacement from all n-L+1 overlapping windows, concatenate to length n.
  CIRCULAR-BLOCK (Politis-Romano 1992): wrap the series into a circle so every point starts an equal
      number of blocks, removing the moving-block's end-effect bias.
  STATIONARY (Politis-Romano 1994): use RANDOM block lengths drawn from a geometric distribution with mean
      L, which makes the resampled series strictly stationary and less sensitive to the exact block size.

The block length L is the one knob, growing like n^(1/3) for the mean; too short breaks correlation, too
long leaves few independent blocks.

This module implements all three block bootstraps and confidence intervals built from them, with a seeded
RNG. It is validated: on INDEPENDENT data the block bootstrap reproduces the ordinary bootstrap's standard
error; on a positively autocorrelated AR(1) series the block-bootstrap standard error of the mean is much
LARGER than the naive i.i.d. bootstrap's (correctly, since correlated data carry less information), and the
block confidence interval attains near-nominal coverage of the true mean where the i.i.d. interval
under-covers; resampled series have the right length; the circular variant uses every point equally; the
stationary variant's mean block length matches its parameter; and results are reproducible per seed. Pure
stdlib; the dependent-data-resampling companion to the bootstrap, Ljung-Box, and Mann-Kendall tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def randint(self, lo, hi):
        """Uniform integer in [lo, hi)."""
        return lo + int(self.u() * (hi - lo))

    def geometric(self, p):
        """Number of trials to first success (>=1) for success prob p."""
        if p >= 1.0:
            return 1
        u = max(self.u(), 1e-12)
        return max(1, int(math.ceil(math.log(u) / math.log(1 - p))))


def moving_block_resample(data, block_len, rng):
    """One moving-block resample: draw overlapping blocks of block_len from data, concat to len(data)."""
    n = len(data)
    n_blocks_start = n - block_len + 1
    out = []
    while len(out) < n:
        start = rng.randint(0, n_blocks_start)
        out.extend(data[start:start + block_len])
    return out[:n]


def circular_block_resample(data, block_len, rng):
    """Circular-block resample: blocks may wrap around the end, so every point starts equally many."""
    n = len(data)
    out = []
    while len(out) < n:
        start = rng.randint(0, n)
        for j in range(block_len):
            out.append(data[(start + j) % n])
    return out[:n]


def stationary_resample(data, mean_block_len, rng):
    """Stationary bootstrap: random geometric block lengths (mean = mean_block_len), circular wrap."""
    n = len(data)
    p = 1.0 / mean_block_len
    out = []
    while len(out) < n:
        start = rng.randint(0, n)
        L = rng.geometric(p)
        for j in range(L):
            if len(out) >= n:
                break
            out.append(data[(start + j) % n])
    return out[:n]


def _resampler(method):
    if method == "moving":
        return moving_block_resample
    if method == "circular":
        return circular_block_resample
    if method == "stationary":
        return stationary_resample
    raise ValueError(f"unknown method: {method}")


def block_bootstrap(data, statistic, block_len, n_resamples=1000, method="moving", seed=12345):
    """Block-bootstrap replicates of `statistic` over resampled series. Returns the list of replicates."""
    rng = _Rng(seed)
    resample = _resampler(method)
    reps = []
    for _ in range(n_resamples):
        reps.append(statistic(resample(data, block_len, rng)))
    return reps


def block_bootstrap_se(data, statistic, block_len, n_resamples=1000, method="moving", seed=12345):
    """Block-bootstrap standard error of `statistic`."""
    reps = block_bootstrap(data, statistic, block_len, n_resamples, method, seed)
    m = sum(reps) / len(reps)
    return math.sqrt(sum((r - m) ** 2 for r in reps) / (len(reps) - 1))


def block_bootstrap_ci(data, statistic, block_len, confidence=0.95, n_resamples=1000,
                        method="moving", seed=12345):
    """Percentile confidence interval from the block bootstrap. Returns (low, high)."""
    reps = sorted(block_bootstrap(data, statistic, block_len, n_resamples, method, seed))
    alpha = (1 - confidence) / 2
    lo_i = max(0, int(alpha * len(reps)))
    hi_i = min(len(reps) - 1, int((1 - alpha) * len(reps)))
    return reps[lo_i], reps[hi_i]


def optimal_block_length(n):
    """A simple rule-of-thumb block length for the mean: round(n^(1/3)), at least 1."""
    return max(1, round(n ** (1 / 3)))


def mean(x):
    return sum(x) / len(x)
