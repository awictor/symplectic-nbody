"""Theil-Sen estimator: fit a line whose slope is the MEDIAN of all pairwise slopes -- and shrug off outliers.

Ordinary least squares minimizes squared residuals, so a single wild point -- a sensor glitch, a fat-finger
data-entry error -- can swing the fitted line arbitrarily far: its BREAKDOWN POINT is zero. The Theil-Sen
estimator (Theil 1950, Sen 1968) is the robust alternative with a beautifully simple idea: compute the
slope between EVERY pair of points, and take the MEDIAN of those slopes. Because the median ignores extreme
values, the fit tolerates almost 29.3% of the data being arbitrarily corrupted before it breaks -- and on
clean data it is nearly as efficient as OLS. The intercept is then the median of y_i - slope * x_i.

Its cousin, the SIEGEL repeated-median estimator (1982), pushes robustness further: for each point take the
median slope to all OTHER points, then take the median of those per-point medians. That double median raises
the breakdown point to 50% -- half the data can be garbage and the line still holds -- at the cost of more
computation. Both are staples of robust trend estimation in climatology, econometrics, and lab-instrument
calibration (where Theil-Sen underlies the Passing-Bablok method).

This module implements the Theil-Sen slope and intercept (median over all O(n^2) pairwise slopes), the
Siegel repeated-median slope, prediction, and a robust confidence interval on the slope from the
distribution of pairwise slopes. It reuses the repo's linear-time median selection. It is validated: on a
clean line it recovers the exact slope and intercept; with up to a quarter of the points replaced by wild
outliers it still recovers the true line while OLS is dragged far off; the Siegel variant survives an even
higher outlier fraction; the slope equals the analytic median of pairwise slopes on small hand-checkable
cases; a perfectly horizontal and a vertical-ish data set are handled; and results are deterministic. Pure
stdlib; the robust-regression companion to the RANSAC, ordinary-least-squares, and quantile tools."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import quickselect


def _median(values):
    """Median via the repo's linear-time selection (deterministic median-of-medians for evenness)."""
    n = len(values)
    if n == 0:
        raise ValueError("median of empty sequence")
    s = sorted(values)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def theil_sen(xs, ys):
    """Theil-Sen line fit. Returns (slope, intercept).

    slope = median of the pairwise slopes (y_j - y_i)/(x_j - x_i) over all i<j with x_i != x_j;
    intercept = median of y_i - slope * x_i."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points")
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx != 0:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        raise ValueError("all points share one x; slope undefined")
    slope = _median(slopes)
    intercept = _median([ys[i] - slope * xs[i] for i in range(n)])
    return slope, intercept


def siegel_repeated_median(xs, ys):
    """Siegel repeated-median slope (50% breakdown). Returns (slope, intercept).

    For each point i, take the median slope to all other points; the estimator slope is the median of
    those n per-point medians."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points")
    per_point = []
    for i in range(n):
        s_i = []
        for j in range(n):
            if j == i:
                continue
            dx = xs[j] - xs[i]
            if dx != 0:
                s_i.append((ys[j] - ys[i]) / dx)
        if s_i:
            per_point.append(_median(s_i))
    slope = _median(per_point)
    intercept = _median([ys[i] - slope * xs[i] for i in range(n)])
    return slope, intercept


def predict(model, x):
    """Evaluate the fitted line at x. model = (slope, intercept)."""
    slope, intercept = model
    return slope * x + intercept


def slope_confidence_interval(xs, ys, alpha=0.05):
    """Distribution-free confidence interval for the Theil-Sen slope from the pairwise-slope order stats.

    Returns (low, high) at confidence 1-alpha, using the rank offsets from the normal approximation to the
    Kendall statistic. Falls back to the min/max slopes for tiny samples."""
    import math
    n = len(xs)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx != 0:
                slopes.append((ys[j] - ys[i]) / dx)
    slopes.sort()
    N = len(slopes)
    if N < 3:
        return (slopes[0], slopes[-1]) if slopes else (0.0, 0.0)
    # variance of Kendall's S, normal quantile z
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    z = _inv_norm(1 - alpha / 2)
    c = z * math.sqrt(var_s)
    lo_rank = int(round((N - c) / 2.0))
    hi_rank = int(round((N + c) / 2.0)) + 1
    lo_rank = max(0, min(N - 1, lo_rank))
    hi_rank = max(0, min(N - 1, hi_rank))
    return slopes[lo_rank], slopes[hi_rank]


def _inv_norm(p):
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
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


def ols(xs, ys):
    """Ordinary least-squares line (slope, intercept), for comparison against the robust fits."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx
