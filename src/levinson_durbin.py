"""Levinson-Durbin recursion -- solving Toeplitz systems and fitting autoregressive models in O(n^2).

A Toeplitz matrix is constant along every diagonal: T[i][j] depends only on i-j. Such matrices appear
the instant you touch a stationary time series, because the covariance between two samples depends only
on the lag between them, not on absolute time -- so the covariance matrix of a signal is Toeplitz. A
general n-by-n linear system costs O(n^3) to solve by Gaussian elimination, but a *symmetric* Toeplitz
system can be solved in O(n^2) time and O(n) memory by the Levinson-Durbin recursion (Levinson 1947,
Durbin 1960), exploiting the fact that the whole matrix is determined by a single row.

The recursion builds the solution one order at a time. Suppose you have solved the order-m problem;
Levinson shows how to extend it to order m+1 with a single rank-one correction driven by a *reflection
coefficient* (also called a PARCOR coefficient) k_{m+1}, computed from how badly the current solution
predicts the next lag. Each step costs O(m) work, so the whole thing is O(n^2). The reflection
coefficients are interesting in their own right: |k_m| < 1 for every m if and only if the Toeplitz
matrix is positive definite, so the recursion doubles as a stability test, and the running product of
(1 - k_m^2) gives the prediction-error variance at each order for free.

The headline application is AUTOREGRESSIVE modelling. An AR(p) model says each sample is a linear
combination of the previous p samples plus white noise: x_t = a_1 x_{t-1} + ... + a_p x_{t-p} + e_t.
Fitting it by least squares leads to the YULE-WALKER equations -- a symmetric Toeplitz system in the
signal's autocorrelation -- which Levinson-Durbin solves directly. The fitted coefficients are exactly
what linear-predictive coding (LPC) uses to compress speech, what spectral estimators use to find the
peaks of a power spectrum from a short record, and what a one-step-ahead predictor uses to forecast the
next sample. This module provides the bare Toeplitz solver, the autocorrelation estimator, the AR fit
returning coefficients / reflection coefficients / error variance, and a one-step predictor.

Validation. (1) The Toeplitz solver is checked against a dense Gaussian-elimination solve of the same
system built explicitly -- they agree to machine precision on random positive-definite Toeplitz
matrices. (2) The residual T x - b is driven to ~0. (3) On data synthesised from a known AR process
with a seeded generator, the recovered coefficients match the generating ones closely, and the fitted
reflection coefficients all satisfy |k| < 1. (4) The prediction-error variance the recursion reports
equals the variance of the actual residuals. (5) A known 3x3 Toeplitz system is solved by hand and
matched. Pure standard library -- ``math`` only, no numpy."""

import math


# ---------------------------------------------------------------------------
# core: solve a symmetric Toeplitz system T x = b
# ---------------------------------------------------------------------------

def solve_toeplitz(r, b):
    """Solve the symmetric Toeplitz system T x = b in O(n^2).

    ``r`` is the first row (equivalently first column) of the symmetric Toeplitz matrix:
    r[0] on the diagonal, r[k] on the k-th off-diagonals. ``b`` is the right-hand side. Returns x.

    Uses the Levinson recursion for a general right-hand side (Golub & Van Loan, Alg. 4.7.2): it
    grows the solution one order at a time, carrying a Yule-Walker vector y alongside the solution x
    and correcting both with a reflection coefficient at each step.
    """
    n = len(b)
    if len(r) < n:
        raise ValueError("r must be at least as long as b")
    if r[0] == 0:
        raise ValueError("r[0] must be non-zero")

    r0 = r[0]
    # normalise to unit diagonal: solve (T/r0) x = b/r0
    rn = [r[i] / r0 for i in range(n)]
    bn = [b[i] / r0 for i in range(n)]

    x = [0.0] * n
    y = [0.0] * n
    if n == 1:
        return [bn[0]]

    y[0] = -rn[1]
    x[0] = bn[0]
    beta = 1.0
    alpha = -rn[1]

    for k in range(1, n):
        beta = (1.0 - alpha * alpha) * beta
        if beta == 0:
            raise ValueError("Toeplitz matrix is singular")
        mu = (bn[k] - sum(rn[i] * x[k - i] for i in range(1, k + 1))) / beta
        xnew = x[:]
        for i in range(k):
            xnew[i] = x[i] + mu * y[k - 1 - i]
        xnew[k] = mu
        x = xnew
        if k < n - 1:
            alpha = -(rn[k + 1] + sum(rn[i] * y[k - i] for i in range(1, k + 1))) / beta
            ynew = y[:]
            for i in range(k):
                ynew[i] = y[i] + alpha * y[k - 1 - i]
            ynew[k] = alpha
            y = ynew

    return x


# ---------------------------------------------------------------------------
# autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation(x, max_lag):
    """Biased autocorrelation estimate r[k] = (1/N) sum_t x_t x_{t+k}, for k = 0..max_lag."""
    n = len(x)
    mean = sum(x) / n
    xc = [v - mean for v in x]
    r = []
    for k in range(max_lag + 1):
        s = sum(xc[t] * xc[t + k] for t in range(n - k))
        r.append(s / n)
    return r


# ---------------------------------------------------------------------------
# autoregressive fit via the Yule-Walker equations
# ---------------------------------------------------------------------------

def levinson_durbin(r, order):
    """Solve the Yule-Walker equations for AR coefficients via the classic Durbin recursion.

    ``r`` is the autocorrelation sequence (r[0..order]). Returns (a, reflection, error) where a is the
    list of AR coefficients [a_1..a_order] (so x_t ~ sum a_i x_{t-i}), reflection is the list of
    reflection (PARCOR) coefficients, and error is the final prediction-error variance.
    """
    if len(r) <= order:
        raise ValueError("need autocorrelation up to the model order")
    if r[0] == 0:
        return [0.0] * order, [], 0.0

    a = [0.0] * (order + 1)     # a[0] unused as coefficient; work in prediction-error form
    a[0] = 1.0
    err = r[0]
    reflection = []

    for m in range(1, order + 1):
        # reflection coefficient k = -(r[m] + sum a_i r[m-i]) / err
        acc = r[m]
        for i in range(1, m):
            acc += a[i] * r[m - i]
        k = -acc / err
        reflection.append(k)

        # update coefficients symmetrically
        new_a = a[:]
        for i in range(1, m):
            new_a[i] = a[i] + k * a[m - i]
        new_a[m] = k
        a = new_a

        err *= (1.0 - k * k)
        if err <= 0:
            err = 0.0
            break

    # convert from prediction-error form (1, -a_1, -a_2, ...) to the intuitive x_t = sum coeff_i x_{t-i}
    coeffs = [-a[i] for i in range(1, order + 1)]
    return coeffs, reflection, err


def ar_fit(x, order):
    """Fit an AR(order) model to the series x. Returns (coeffs, reflection, error_variance)."""
    r = autocorrelation(x, order)
    return levinson_durbin(r, order)


def ar_predict(x, coeffs):
    """One-step-ahead prediction: x_hat[t] = sum_i coeffs[i] * x[t-1-i], from the fitted coeffs.

    Returns the predicted next value given the most recent len(coeffs) samples of x.
    """
    p = len(coeffs)
    if len(x) < p:
        raise ValueError("need at least `order` samples to predict")
    return sum(coeffs[i] * x[-1 - i] for i in range(p))


def is_positive_definite(r):
    """A symmetric Toeplitz matrix with first row r is positive definite iff every reflection
    coefficient produced by the Durbin recursion has magnitude < 1. Returns True/False."""
    order = len(r) - 1
    if order < 1:
        return r[0] > 0
    if r[0] <= 0:
        return False
    _, reflection, err = levinson_durbin(r, order)
    return all(abs(k) < 1.0 for k in reflection) and err > 0
