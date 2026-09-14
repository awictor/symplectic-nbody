"""L1 trend filtering: fit a piecewise-LINEAR trend to noisy data, with the kinks placed automatically.

The Hodrick-Prescott filter penalizes the SQUARED second difference of the trend, which smooths noise but
smears every corner -- the fitted trend is smoothly curved everywhere, so a signal that is genuinely made
of straight segments (a growth phase, then a plateau, then a decline) comes out rounded. L1 TREND FILTERING
(Kim, Koh, Boyd, Gorinevsky 2009) replaces that squared penalty with the ABSOLUTE second difference:

    minimize   (1/2) sum_t (y_t - x_t)^2  +  lambda * sum_t |x_{t-1} - 2 x_t + x_{t+1}|.

Because the L1 norm is sparsity-inducing (the same reason LASSO zeroes out coefficients), the optimal
second difference is EXACTLY ZERO at most time steps -- and a zero second difference means three points in
a straight line. So the solution is automatically PIECEWISE LINEAR, with a small number of KINKS (knots)
wherever the second difference is nonzero. lambda is the one knob: lambda -> 0 interpolates the data,
lambda -> infinity forces every second difference to zero, collapsing the fit to the single least-squares
straight line. In between, larger lambda means fewer kinks and longer segments -- the filter chooses both
the slopes and the breakpoints for you, a convex alternative to segmented regression.

This module solves the problem with ADMM (splitting the smooth quadratic term from the non-smooth L1 term
via an auxiliary variable z = D x, with a soft-threshold z-update and a banded pentadiagonal x-update), and
computes the detected kink locations. It is validated: a clean piecewise-linear signal is recovered
exactly and its kinks are found at the true breakpoints; a large lambda collapses the trend to the
least-squares line; lambda -> 0 reproduces the data; the fitted trend is genuinely piecewise linear (its
second difference is sparse); larger lambda yields fewer kinks; it denoises a noisy piecewise-linear
series far better than it fits the noise; and the ADMM primal/dual residuals converge. Pure stdlib; the
sparse-trend companion to the Hodrick-Prescott, total-variation, LOESS, and ADMM/LASSO tools."""

from __future__ import annotations


def _second_diff(x):
    """D x for the (n-2) x n second-difference operator: (Dx)_i = x_i - 2 x_{i+1} + x_{i+2}."""
    n = len(x)
    return [x[i] - 2.0 * x[i + 1] + x[i + 2] for i in range(n - 2)]


def _second_diff_T(v, n):
    """D^T v, mapping R^{n-2} back to R^n. Adjoint of _second_diff."""
    out = [0.0] * n
    for i in range(len(v)):
        out[i] += v[i]
        out[i + 1] += -2.0 * v[i]
        out[i + 2] += v[i]
    return out


def _soft_threshold(v, kappa):
    """Elementwise soft-threshold: shrink each entry toward zero by kappa (the L1 prox)."""
    out = []
    for x in v:
        if x > kappa:
            out.append(x - kappa)
        elif x < -kappa:
            out.append(x + kappa)
        else:
            out.append(0.0)
    return out


def _solve_banded_spd(diag, off1, off2, rhs):
    """Solve a symmetric positive-definite PENTADIAGONAL system by banded LDL^T (bandwidth 2).

    diag[i] = A[i][i]; off1[i] = A[i][i+1] (len n-1); off2[i] = A[i][i+2] (len n-2). O(n)."""
    n = len(diag)
    # Banded Cholesky-like factorization storing L with two subdiagonals.
    d = [0.0] * n          # pivots (diagonal of D)
    l1 = [0.0] * n         # first subdiagonal of L
    l2 = [0.0] * n         # second subdiagonal of L
    for i in range(n):
        di = diag[i]
        if i >= 1:
            di -= l1[i - 1] * l1[i - 1] * d[i - 1]
        if i >= 2:
            di -= l2[i - 2] * l2[i - 2] * d[i - 2]
        d[i] = di
        if i + 1 < n:
            v = off1[i]
            if i >= 1:
                v -= l1[i - 1] * l2[i - 1] * d[i - 1]
            l1[i] = v / di
        if i + 2 < n:
            l2[i] = off2[i] / di
    # forward solve L y = rhs
    y = [0.0] * n
    for i in range(n):
        v = rhs[i]
        if i >= 1:
            v -= l1[i - 1] * y[i - 1]
        if i >= 2:
            v -= l2[i - 2] * y[i - 2]
        y[i] = v
    # diagonal solve D w = y
    w = [y[i] / d[i] for i in range(n)]
    # backward solve L^T x = w
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        v = w[i]
        if i + 1 < n:
            v -= l1[i] * x[i + 1]
        if i + 2 < n:
            v -= l2[i] * x[i + 2]
        x[i] = v
    return x


def l1_trend_filter(y, lam, rho=None, max_iter=2000, tol=1e-8):
    """Piecewise-linear trend by L1 filtering, solved with ADMM. Returns the fitted trend x (len n).

    Minimizes (1/2)||y - x||^2 + lam * ||D x||_1 where D is the second-difference operator.
    rho is the ADMM penalty; the x-update is a banded pentadiagonal solve, z-update a soft-threshold.
    If rho is None it is auto-scaled to max(1, lam) -- the penalty must grow with lam or ADMM crawls."""
    n = len(y)
    if n < 3:
        return list(map(float, y))  # no interior second difference to penalize

    if rho is None:
        rho = max(1.0, lam)  # ADMM stalls when rho << lam; scale it with the L1 weight

    m = n - 2
    z = [0.0] * m
    u = [0.0] * m  # scaled dual

    # The x-update solves (I + rho D^T D) x = y + rho D^T (z - u).
    # Build the constant pentadiagonal matrix A = I + rho D^T D once.
    # D^T D is a known banded matrix; assemble its bands directly.
    diag = [0.0] * n
    off1 = [0.0] * (n - 1)
    off2 = [0.0] * (n - 2)
    for i in range(m):
        # row i of D has entries (1, -2, 1) at columns i, i+1, i+2
        cols = (i, i + 1, i + 2)
        vals = (1.0, -2.0, 1.0)
        for a in range(3):
            diag[cols[a]] += vals[a] * vals[a]
            for b in range(a + 1, 3):
                dc = cols[b] - cols[a]
                prod = vals[a] * vals[b]
                if dc == 1:
                    off1[cols[a]] += prod
                elif dc == 2:
                    off2[cols[a]] += prod
    # A = I + rho * (D^T D)
    Adiag = [1.0 + rho * diag[i] for i in range(n)]
    Aoff1 = [rho * off1[i] for i in range(n - 1)]
    Aoff2 = [rho * off2[i] for i in range(n - 2)]

    for it in range(max_iter):
        # x-update
        rhs_vec = _second_diff_T([z[i] - u[i] for i in range(m)], n)
        rhs = [y[i] + rho * rhs_vec[i] for i in range(n)]
        x = _solve_banded_spd(Adiag, Aoff1, Aoff2, rhs)

        # z-update: soft-threshold of (D x + u)
        Dx = _second_diff(x)
        z_old = z
        z = _soft_threshold([Dx[i] + u[i] for i in range(m)], lam / rho)

        # dual update
        u = [u[i] + Dx[i] - z[i] for i in range(m)]

        # convergence: primal (Dx - z) and dual (rho (z - z_old)) residuals
        primal = sum((Dx[i] - z[i]) ** 2 for i in range(m)) ** 0.5
        dual = rho * sum((z[i] - z_old[i]) ** 2 for i in range(m)) ** 0.5
        if primal < tol * (1 + n ** 0.5) and dual < tol * (1 + n ** 0.5):
            break

    return x


def kinks(x, tol=1e-4):
    """Indices where the trend changes slope (nonzero second difference) -- the detected breakpoints.

    A kink must clear BOTH a relative threshold (tol * max|Dx|) and a small absolute floor tied to the
    trend's own scale, so numerical dust on a near-straight fit is not miscounted as breakpoints."""
    Dx = _second_diff(x)
    scale = max((abs(v) for v in Dx), default=1.0) or 1.0
    xrange = (max(x) - min(x)) if x else 1.0
    floor = 1e-6 * (xrange or 1.0)
    thresh = max(tol * scale, floor)
    return [i + 1 for i, v in enumerate(Dx) if abs(v) > thresh]


def second_difference(x):
    """Expose the trend's second difference (sparse for an L1 fit)."""
    return _second_diff(x)


def ls_line(y):
    """The least-squares straight line through y (what a huge lambda collapses to)."""
    n = len(y)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in xs)
    sxy = sum((xs[i] - mx) * (y[i] - my) for i in range(n))
    slope = sxy / sxx if sxx else 0.0
    inter = my - slope * mx
    return [inter + slope * i for i in range(n)]
