"""RANSAC: fitting a model to data riddled with outliers by consensus, not least squares.

Least squares fits every point, so a handful of gross OUTLIERS -- mismeasurements, wrong
correspondences, points from a different object -- drag the fit arbitrarily far from the truth. One
bad point can ruin an entire regression. RANSAC (RANdom SAmple Consensus, Fischler and Bolles, 1981)
takes the opposite stance: assume most of the data is good ("inliers") and the rest is noise to be
ignored. It repeatedly

    1. draws the MINIMAL random sample needed to define the model (2 points for a line, 3 for a
       circle),
    2. fits the model to just that sample,
    3. counts how many of ALL the points agree with it to within a tolerance (the CONSENSUS SET),

and keeps the model with the largest consensus. A final least-squares refit on the winning inliers
polishes the estimate. Because a single clean sample is enough to reveal the right model, RANSAC
tolerates outlier fractions that destroy least squares -- even a majority of outliers, given enough
iterations.

How many iterations? If a fraction w of points are inliers and the sample needs s points, a single
sample is all-inlier with probability w^s, so to succeed with probability p the count is
N = log(1 - p) / log(1 - w^s). This module computes that adaptively (shrinking N as a better
consensus raises the inlier estimate), and ships robust LINE and CIRCLE fitters plus a generic RANSAC
driver you hand a sampler, a fitter, and a residual function.

Validated on synthetic data with known ground truth: on points from a known line (or circle) plus
heavy uniform outlier contamination, RANSAC recovers the true parameters to tolerance while ordinary
least squares is dragged far off; the recovered inlier set matches the planted inliers; and the
adaptive iteration count matches the closed-form formula. Deterministic under a seed. Pure stdlib;
the robust-estimation companion to the least-squares and Theil-Sen-style fits."""

from __future__ import annotations

import math


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def required_iterations(inlier_fraction, sample_size, success_prob=0.99):
    """N = log(1 - p) / log(1 - w^s): iterations to see one all-inlier sample with prob >= p."""
    w = inlier_fraction
    if w <= 0:
        return float("inf")
    denom = 1 - w ** sample_size
    if denom <= 0:
        return 1
    return math.log(1 - success_prob) / math.log(denom)


def ransac(points, sample_size, fit_fn, residual_fn, threshold,
           max_iterations=1000, success_prob=0.99, seed=12345, min_inliers=None):
    """Generic RANSAC.

    points: list of data items.
    sample_size: minimal points to fit the model.
    fit_fn(sample) -> model or None (None if the sample is degenerate).
    residual_fn(model, point) -> non-negative error.
    threshold: a point is an inlier if residual <= threshold.
    Returns dict with best model, inlier indices, and iterations actually run."""
    n = len(points)
    if n < sample_size:
        raise ValueError("not enough points for a sample")
    rng = _lcg(seed)
    best_model = None
    best_inliers = []
    iters_run = 0
    N = max_iterations

    i = 0
    while i < min(N, max_iterations):
        i += 1
        iters_run = i
        sample = _sample_indices(n, sample_size, rng)
        model = fit_fn([points[j] for j in sample])
        if model is None:
            continue
        inliers = [k for k in range(n) if residual_fn(model, points[k]) <= threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = model
            # adaptively shrink the needed iterations as the inlier estimate improves
            w = len(inliers) / n
            if w > 0:
                need = required_iterations(w, sample_size, success_prob)
                N = min(N, need)

    if min_inliers is not None and len(best_inliers) < min_inliers:
        return {"model": None, "inliers": [], "iterations": iters_run}
    return {"model": best_model, "inliers": best_inliers, "iterations": iters_run}


def _sample_indices(n, k, rng):
    chosen = set()
    while len(chosen) < k:
        chosen.add(rng() % n)
    return list(chosen)


# --- line fitting ------------------------------------------------------------
def fit_line(sample):
    """Fit a line through 2 points as (a, b, c) with a*x + b*y + c = 0, normalized so a^2+b^2=1.
    Returns None if the two points coincide."""
    (x1, y1), (x2, y2) = sample[0], sample[1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    # normal to the direction (dx,dy) is (dy,-dx)
    a, b = dy / norm, -dx / norm
    c = -(a * x1 + b * y1)
    return (a, b, c)


def line_residual(model, point):
    """Perpendicular distance from a point to the line a*x+b*y+c=0 (a^2+b^2=1)."""
    a, b, c = model
    x, y = point
    return abs(a * x + b * y + c)


def refit_line_lsq(points):
    """Total-least-squares line fit (principal axis) of a set of points -> (a, b, c)."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    # direction is the top eigenvector of the covariance; normal is the smaller one
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    dx, dy = math.cos(theta), math.sin(theta)
    a, b = dy, -dx
    c = -(a * mx + b * my)
    return (a, b, c)


def ransac_line(points, threshold, **kwargs):
    """RANSAC line fit with a least-squares refit on the consensus set."""
    res = ransac(points, 2, fit_line, line_residual, threshold, **kwargs)
    if res["model"] is not None and len(res["inliers"]) >= 2:
        res["model"] = refit_line_lsq([points[i] for i in res["inliers"]])
    return res


# --- circle fitting ----------------------------------------------------------
def fit_circle(sample):
    """Fit a circle through 3 points -> (cx, cy, r). None if the points are collinear."""
    (x1, y1), (x2, y2), (x3, y3) = sample[0], sample[1], sample[2]
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return None
    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) +
          (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) +
          (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    return (ux, uy, r)


def circle_residual(model, point):
    """Distance from a point to the circle: |dist(point, center) - r|."""
    cx, cy, r = model
    return abs(math.hypot(point[0] - cx, point[1] - cy) - r)


def ransac_circle(points, threshold, **kwargs):
    """RANSAC circle fit."""
    return ransac(points, 3, fit_circle, circle_residual, threshold, **kwargs)


# --- ordinary least squares (for comparison) ---------------------------------
def ols_line(points):
    """Ordinary vertical-offset least-squares line y = m x + b as (a,b,c) with a=-m,b=1,c=-b0,
    normalized. Sensitive to outliers -- included to contrast with RANSAC."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    if sxx == 0:
        return (1.0, 0.0, -mx)
    m = sxy / sxx
    b0 = my - m * mx
    # y = m x + b0  ->  m x - y + b0 = 0
    norm = math.hypot(m, 1)
    return (m / norm, -1 / norm, b0 / norm)
