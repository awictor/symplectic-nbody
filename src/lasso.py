"""Lasso regression by coordinate descent: least squares that selects features by zeroing them out.

Ordinary least squares fits every feature; when there are many features and few of them matter, it
overfits and gives dense, uninterpretable coefficients. RIDGE regression adds an L2 penalty
lambda*||w||^2 that shrinks coefficients toward zero but never exactly to zero. The LASSO (Tibshirani
1996) uses an L1 penalty lambda*||w||_1 instead, and that one change is transformative: the L1 penalty
drives most coefficients to EXACTLY zero, so Lasso performs automatic FEATURE SELECTION -- it picks a
sparse subset of predictors and discards the rest. It is the standard tool when you believe only a
handful of many candidate variables truly drive the response.

The objective, (1/2n)||y - Xw||^2 + lambda*||w||_1, is convex but non-differentiable at zero, so
gradient descent alone won't do. COORDINATE DESCENT solves it beautifully: cycle through the
coefficients one at a time, and for each, the optimal update (holding the others fixed) is the
SOFT-THRESHOLD of the least-squares coordinate value,

    w_j = soft_threshold( (1/n) x_j^T r_j , lambda ) / (x_j^T x_j / n),

where r_j is the residual excluding feature j and soft_threshold(z, t) = sign(z) * max(|z| - t, 0)
pulls z toward zero and clamps it there once it is smaller than t. Cycling to convergence gives the
exact Lasso solution. This module standardizes the features, runs coordinate descent (with an
unpenalized intercept), and computes the coefficient path over a range of lambda.

Validated: on data generated from a known sparse coefficient vector, Lasso recovers the correct
support (zeros the irrelevant features) and estimates the true nonzero coefficients closely; a larger
lambda yields a sparser model (more zeros); lambda = 0 reduces to ordinary least squares; the sparsity
increases monotonically along the lambda path; and predictions are accurate on the active features. It
is contrasted with a ridge fit, which keeps every coefficient nonzero. Pure stdlib; the
sparse-regression companion to the ordinary/ridge regression and the orthogonal-matching-pursuit
tools."""

from __future__ import annotations


def _soft_threshold(z, t):
    if z > t:
        return z - t
    if z < -t:
        return z + t
    return 0.0


def _standardize(X):
    """Center and scale each column to zero mean, unit variance. Returns (Xs, means, stds)."""
    n = len(X)
    p = len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    stds = []
    for j in range(p):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n
        stds.append(var ** 0.5 if var > 0 else 1.0)
    Xs = [[(X[i][j] - means[j]) / stds[j] for j in range(p)] for i in range(n)]
    return Xs, means, stds


def lasso(X, y, lam, max_iter=1000, tol=1e-7):
    """Fit Lasso regression by coordinate descent. Returns (intercept, coefficients) in the ORIGINAL
    feature scale. lam is the L1 penalty strength."""
    n = len(X)
    p = len(X[0])
    Xs, means, stds = _standardize(X)
    ybar = sum(y) / n
    yc = [y[i] - ybar for i in range(n)]  # centered target; intercept handled separately

    w = [0.0] * p
    # precompute column norms (in standardized space each column has variance 1, so norm^2 = n)
    col_sq = [sum(Xs[i][j] ** 2 for i in range(n)) for j in range(p)]

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            # residual excluding feature j
            # r_j = yc - Xs w + Xs[:,j] w_j
            rho = 0.0
            for i in range(n):
                pred = sum(Xs[i][k] * w[k] for k in range(p))
                r_ij = yc[i] - pred + Xs[i][j] * w[j]
                rho += Xs[i][j] * r_ij
            old = w[j]
            w[j] = _soft_threshold(rho, lam * n) / col_sq[j] if col_sq[j] > 0 else 0.0
            max_change = max(max_change, abs(w[j] - old))
        if max_change < tol:
            break

    # unstandardize: w_orig_j = w_j / std_j; intercept = ybar - sum(w_orig_j * mean_j)
    coef = [w[j] / stds[j] for j in range(p)]
    intercept = ybar - sum(coef[j] * means[j] for j in range(p))
    return intercept, coef


def predict(intercept, coef, X):
    """Predict y for each row of X."""
    return [intercept + sum(coef[j] * X[i][j] for j in range(len(coef))) for i in range(len(X))]


def lasso_path(X, y, lambdas, **kwargs):
    """Fit Lasso for each lambda; returns a list of (lambda, intercept, coef, n_nonzero)."""
    out = []
    for lam in lambdas:
        intercept, coef = lasso(X, y, lam, **kwargs)
        nz = sum(1 for c in coef if abs(c) > 1e-8)
        out.append((lam, intercept, coef, nz))
    return out


def n_nonzero(coef, tol=1e-8):
    return sum(1 for c in coef if abs(c) > tol)


def mse(intercept, coef, X, y):
    preds = predict(intercept, coef, X)
    return sum((preds[i] - y[i]) ** 2 for i in range(len(y))) / len(y)


# --- ridge, for contrast (keeps all coefficients nonzero) --------------------
def ridge(X, y, lam):
    """Ridge regression (L2) via the normal equations (X^T X + lam I) w = X^T y, with an intercept.
    Returns (intercept, coefficients). Included to contrast with Lasso's sparsity."""
    n = len(X)
    p = len(X[0])
    # center
    ybar = sum(y) / n
    means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    Xc = [[X[i][j] - means[j] for j in range(p)] for i in range(n)]
    yc = [y[i] - ybar for i in range(n)]
    # (Xc^T Xc + lam I) w = Xc^T yc
    A = [[sum(Xc[i][a] * Xc[i][b] for i in range(n)) + (lam if a == b else 0)
          for b in range(p)] for a in range(p)]
    rhs = [sum(Xc[i][a] * yc[i] for i in range(n)) for a in range(p)]
    w = _solve(A, rhs)
    intercept = ybar - sum(w[j] * means[j] for j in range(p))
    return intercept, w


def _solve(M, b):
    n = len(b)
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        p = A[col][col]
        if abs(p) < 1e-15:
            continue
        for r in range(col + 1, n):
            f = A[r][col] / p
            for c in range(col, n + 1):
                A[r][c] -= f * A[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-15:
            continue
        s = A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / A[i][i]
    return x
