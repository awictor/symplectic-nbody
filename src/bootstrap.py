"""Bootstrap: confidence intervals from resampling, no formulas required.

Classical confidence intervals need a formula for the sampling distribution of your statistic -- easy
for the mean, hard for the median, a correlation, a ratio, or a trimmed mean. The BOOTSTRAP (Efron,
1979) sidesteps all of that with one idea: the sample itself is the best available stand-in for the
population, so RESAMPLE it with replacement many times, recompute the statistic on each resample, and
the spread of those bootstrap replicates approximates the statistic's true sampling distribution. From
that empirical distribution you read off standard errors and confidence intervals for ANY statistic,
however complicated, without ever deriving its variance. It is one of the most important ideas in
modern statistics precisely because it is universal and assumption-light.

Two intervals are provided. The PERCENTILE interval simply takes the 2.5th and 97.5th percentiles of
the bootstrap replicates (for a 95% CI) -- intuitive and often good enough. The BCa (bias-corrected and
accelerated) interval is the refined version that corrects for two biases the percentile method
ignores: a MEDIAN BIAS (the fraction of replicates below the observed estimate, giving a bias-correction
z0) and ACCELERATION (skewness of the sampling distribution, estimated by jackknife -- leaving each
point out in turn). BCa shifts the percentiles it takes according to z0 and the acceleration, giving
intervals with much better coverage for skewed statistics. The JACKKNIFE itself -- the n leave-one-out
estimates -- also yields a classic bias and standard-error estimate.

This module computes bootstrap replicates (with a seeded reproducible resampler), percentile and BCa
confidence intervals, the bootstrap standard error, and the jackknife bias/standard-error. It is
verified against known quantities -- for the mean of normal data the bootstrap standard error matches
the analytic s/sqrt(n) and the interval brackets the true mean; the jackknife is exact for the mean;
the interval is contained in the data range; and, over many seeded datasets, a 90% interval covers the
true parameter close to 90% of the time. Pure stdlib; a statistics companion to the KS-test, Welford,
and Shannon-entropy notes."""

from __future__ import annotations

import math


class _LCG:
    """Seeded RNG for reproducible resampling."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 8

    def randint(self, n):
        return self.next() % n


def bootstrap_replicates(data, statistic, n_resamples=2000, seed=12345):
    """Compute `statistic` on `n_resamples` bootstrap resamples (drawn with replacement from `data`).
    Returns the sorted list of replicate values."""
    n = len(data)
    rng = _LCG(seed)
    reps = []
    for _ in range(n_resamples):
        resample = [data[rng.randint(n)] for _ in range(n)]
        reps.append(statistic(resample))
    reps.sort()
    return reps


def _percentile(sorted_vals, q):
    """The q-quantile (0..1) of a sorted list by linear interpolation."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def percentile_ci(data, statistic, confidence=0.95, n_resamples=2000, seed=12345):
    """The percentile bootstrap confidence interval for `statistic`. Returns (lo, hi)."""
    reps = bootstrap_replicates(data, statistic, n_resamples, seed)
    alpha = 1 - confidence
    return _percentile(reps, alpha / 2), _percentile(reps, 1 - alpha / 2)


def bootstrap_se(data, statistic, n_resamples=2000, seed=12345):
    """The bootstrap standard error of `statistic`: the standard deviation of the replicates."""
    reps = bootstrap_replicates(data, statistic, n_resamples, seed)
    m = sum(reps) / len(reps)
    var = sum((r - m) ** 2 for r in reps) / (len(reps) - 1) if len(reps) > 1 else 0.0
    return math.sqrt(var)


def _normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _normal_ppf(p):
    """Inverse standard-normal CDF via a rational approximation (Acklam)."""
    if p <= 0:
        return -1e18
    if p >= 1:
        return 1e18
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def jackknife(data, statistic):
    """The jackknife: recompute `statistic` leaving each point out in turn. Returns (estimates,
    bias, standard_error)."""
    n = len(data)
    theta_hat = statistic(data)
    estimates = []
    for i in range(n):
        loo = data[:i] + data[i + 1:]
        estimates.append(statistic(loo))
    mean_jack = sum(estimates) / n
    bias = (n - 1) * (mean_jack - theta_hat)
    se = math.sqrt((n - 1) / n * sum((e - mean_jack) ** 2 for e in estimates))
    return estimates, bias, se


def bca_ci(data, statistic, confidence=0.95, n_resamples=2000, seed=12345):
    """The bias-corrected and accelerated (BCa) bootstrap confidence interval. Returns (lo, hi)."""
    reps = bootstrap_replicates(data, statistic, n_resamples, seed)
    theta_hat = statistic(data)
    n_rep = len(reps)

    # bias-correction z0: how many replicates fall below the observed estimate
    n_below = sum(1 for r in reps if r < theta_hat)
    prop = n_below / n_rep
    if prop <= 0:
        prop = 0.5 / n_rep
    if prop >= 1:
        prop = 1 - 0.5 / n_rep
    z0 = _normal_ppf(prop)

    # acceleration a from the jackknife
    jack, _, _ = jackknife(data, statistic)
    mean_jack = sum(jack) / len(jack)
    num = sum((mean_jack - j) ** 3 for j in jack)
    den = 6.0 * (sum((mean_jack - j) ** 2 for j in jack)) ** 1.5
    a = num / den if den != 0 else 0.0

    alpha = 1 - confidence
    z_lo = _normal_ppf(alpha / 2)
    z_hi = _normal_ppf(1 - alpha / 2)

    def adjust(z):
        denom = 1 - a * (z0 + z)
        if denom == 0:
            denom = 1e-12
        return _normal_cdf(z0 + (z0 + z) / denom)

    return _percentile(reps, adjust(z_lo)), _percentile(reps, adjust(z_hi))


# --- common statistics ------------------------------------------------------
def mean(x):
    return sum(x) / len(x)


def median(x):
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def std(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else 0.0
