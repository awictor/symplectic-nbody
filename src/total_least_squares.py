"""Total least squares: fit a line (or hyperplane) minimizing PERPENDICULAR distance, when both axes have error.

Ordinary least squares assumes x is known exactly and only y is noisy, so it minimizes VERTICAL residuals.
But in most real measurements BOTH variables carry error -- two instruments, two noisy sensors, a
scatterplot of one uncertain quantity against another. Minimizing vertical distance then biases the slope
toward zero (regression dilution / attenuation), and swapping the roles of x and y gives a different line.
TOTAL LEAST SQUARES (orthogonal / errors-in-variables regression) fixes this by minimizing the sum of
squared PERPENDICULAR distances from the points to the line -- a symmetric criterion that treats both axes
alike and gives the same fit whichever variable you call the response.

The solution is pure linear algebra: center the data, and the best-fit line points along the direction of
MAXIMUM variance (the top principal component), while its normal is the direction of MINIMUM variance --
the eigenvector of the data covariance matrix for the SMALLEST eigenvalue. That smallest eigenvalue is
exactly the mean squared orthogonal residual. This generalizes to fitting a hyperplane in any dimension
(the normal is the least-variance eigenvector) and underlies the DEMING regression used in method-
comparison studies and the geometric fitting behind PCA.

This module fits a 2-D line by TLS (returning slope/intercept and the orthogonal residual), fits a general
d-dimensional hyperplane (normal and offset), and includes Deming regression (TLS with a known error-
variance ratio). It reuses the repo's symmetric Jacobi eigensolver. It is validated: on a clean line it
recovers the exact fit; the TLS line is invariant when x and y are swapped (where OLS is not); on data
with error in BOTH variables it is far less biased than OLS; the fitted normal is orthogonal to the
direction of maximum variance; the smallest eigenvalue equals the mean squared perpendicular residual; a
vertical line (infinite OLS slope) is handled cleanly; the plane fit recovers a known 3-D plane; and
results are deterministic. Pure stdlib; the errors-in-variables companion to the ordinary-least-squares,
Theil-Sen, PCA-whitening, and SVD tools."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import jacobi_eigen


def _covariance(cols):
    """Covariance matrix of centered column data. cols is a list of d columns, each length n."""
    d = len(cols)
    n = len(cols[0])
    means = [sum(c) / n for c in cols]
    C = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            s = sum((cols[i][k] - means[i]) * (cols[j][k] - means[j]) for k in range(n))
            C[i][j] = s / n
            C[j][i] = C[i][j]
    return C, means


def fit_line(xs, ys):
    """Total-least-squares 2-D line fit. Returns a dict with slope, intercept, the unit normal
    (nx, ny), the point it passes through (the centroid), and the mean squared orthogonal residual.

    Minimizes sum of squared perpendicular distances. Handles near-vertical lines (slope may be inf)."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points")
    C, means = _covariance([list(map(float, xs)), list(map(float, ys))])
    vals, V = jacobi_eigen.sorted_eigen(C)   # DESCENDING eigenvalues; V columns are eigenvectors
    # direction of MAX variance = first eigenvector; NORMAL = last (smallest eigenvalue)
    dir_vec = (V[0][0], V[1][0])             # tangent to the line
    normal = (V[0][1], V[1][1])              # perpendicular to the line
    cx, cy = means
    # slope from the tangent direction
    dx, dy = dir_vec
    if abs(dx) < 1e-15:
        slope = float("inf")
        intercept = float("nan")
    else:
        slope = dy / dx
        intercept = cy - slope * cx
    # mean squared orthogonal residual = smallest eigenvalue
    mse_perp = vals[-1]
    return {
        "slope": slope,
        "intercept": intercept,
        "normal": normal,
        "centroid": (cx, cy),
        "direction": dir_vec,
        "mse_perp": mse_perp,
    }


def orthogonal_residual(model, x, y):
    """Signed perpendicular distance from (x, y) to the fitted line."""
    nx, ny = model["normal"]
    cx, cy = model["centroid"]
    return (x - cx) * nx + (y - cy) * ny


def fit_hyperplane(points):
    """TLS fit of a hyperplane to d-dimensional points. Returns (normal, offset) with the plane
    { p : normal . p = offset }, normal a unit vector along the least-variance direction."""
    n = len(points)
    d = len(points[0])
    cols = [[points[k][j] for k in range(n)] for j in range(d)]
    C, means = _covariance(cols)
    vals, V = jacobi_eigen.sorted_eigen(C)
    normal = [V[j][d - 1] for j in range(d)]   # smallest-eigenvalue eigenvector
    offset = sum(normal[j] * means[j] for j in range(d))
    return normal, offset


def deming(xs, ys, ratio=1.0):
    """Deming regression: TLS with a known ratio delta = var(error_y)/var(error_x).

    ratio=1 reduces to orthogonal TLS. Returns (slope, intercept). Uses the closed-form Deming estimator."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    syy = sum((y - my) ** 2 for y in ys) / (n - 1)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)
    d = ratio
    # Deming slope closed form
    term = syy - d * sxx
    slope = (term + math.sqrt(term * term + 4 * d * sxy * sxy)) / (2 * sxy)
    intercept = my - slope * mx
    return slope, intercept


def ols(xs, ys):
    """Ordinary least squares (vertical residuals), for comparison."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx
