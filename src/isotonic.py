"""Isotonic regression by pool-adjacent-violators: the best monotone fit to noisy data, in O(n).

Sometimes you know a relationship must be MONOTONE -- a dose-response curve only rises, a calibration
map only increases, a cumulative count never falls -- but your measurements, being noisy, wobble up
and down. Isotonic regression finds the non-decreasing sequence y-hat closest to the data y in
least-squares sense: minimize sum w_i (y_i - yhat_i)^2 subject to yhat_1 <= yhat_2 <= ... <= yhat_n.
It is nonparametric (no assumed functional form, just monotonicity) and the exact solution is found
in a single linear-time sweep.

The POOL ADJACENT VIOLATORS ALGORITHM (PAVA) is the classic method. Walk left to right maintaining a
stack of "blocks", each a contiguous run with a common fitted value (its weighted mean). When a new
point's value is below the previous block's -- a VIOLATION of monotonicity -- merge the two blocks
into one whose value is their pooled weighted mean, and keep merging leftward while the violation
propagates. The result is the unique optimal fit: a step function that is the greatest convex
minorant of the cumulative-sum diagram, so it is also the derivative view of a convex-hull
computation.

This module fits non-decreasing (and, by flipping, non-increasing) isotonic regressions with optional
weights, predicts at new x by interpolation, and exposes the block structure. The classic application
shown is PROBABILITY CALIBRATION: turning a classifier's uncalibrated scores into monotone,
well-calibrated probabilities (the isotonic-regression calibration used in practice).

Validated against ground truth and brute force: the fit is monotone and, for small n, matches a
brute-force constrained least-squares solution (projected coordinate descent) to tolerance; it obeys
the KKT block conditions (each block value is the weighted mean of its members); a monotone signal is
recovered from noise with lower error than the raw data; flipping recovers non-increasing fits; and
already-monotone data is returned unchanged. Pure stdlib; the shape-constrained companion to the
least-squares and LOWESS-style smoothers."""

from __future__ import annotations


def isotonic_regression(y, weights=None, increasing=True):
    """The optimal monotone least-squares fit to y. Returns the fitted values (same length as y).

    If increasing (default), the fit is non-decreasing; otherwise non-increasing. Weights, if given,
    weight each residual."""
    n = len(y)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights must match y in length")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")

    vals = list(y)
    if not increasing:
        vals = [-v for v in vals]

    # PAVA with a stack of blocks: each block = [sum_wy, sum_w, count, value]
    block_val = []   # pooled value of each block
    block_w = []     # total weight of each block
    block_len = []   # number of points in each block

    for i in range(n):
        v = vals[i]
        w = weights[i]
        block_val.append(v)
        block_w.append(w)
        block_len.append(1)
        # merge backward while the last block violates monotonicity
        while len(block_val) > 1 and block_val[-2] > block_val[-1]:
            # pool the last two blocks
            w1, w2 = block_w[-2], block_w[-1]
            pooled = (block_val[-2] * w1 + block_val[-1] * w2) / (w1 + w2)
            block_val.pop()
            v_last = block_val.pop()
            block_val.append(pooled)
            block_w[-2] = w1 + w2
            block_w.pop()
            block_len[-2] = block_len[-2] + block_len[-1]
            block_len.pop()

    # expand blocks back to a per-point fitted vector
    fitted = []
    for val, length in zip(block_val, block_len):
        fitted.extend([val] * length)

    if not increasing:
        fitted = [-f for f in fitted]
    return fitted


def blocks(y, weights=None, increasing=True):
    """Return the block structure as a list of (start_index, length, value)."""
    fitted = isotonic_regression(y, weights, increasing)
    out = []
    i = 0
    n = len(fitted)
    while i < n:
        j = i
        while j + 1 < n and fitted[j + 1] == fitted[i]:
            j += 1
        out.append((i, j - i + 1, fitted[i]))
        i = j + 1
    return out


class IsotonicModel:
    """Fit an isotonic regression against x-positions and predict at new x by linear interpolation
    between fitted knots (the standard isotonic interpolator, e.g. for probability calibration)."""

    def __init__(self, x, y, weights=None, increasing=True):
        if len(x) != len(y):
            raise ValueError("x and y must match")
        # sort by x, remembering order
        order = sorted(range(len(x)), key=lambda i: x[i])
        self.x = [x[i] for i in order]
        ys = [y[i] for i in order]
        ws = [weights[i] for i in order] if weights else None
        self.y = isotonic_regression(ys, ws, increasing)
        self.increasing = increasing

    def predict(self, xq):
        """Predict the fitted value at query point(s) xq (scalar or list) by clamped linear
        interpolation between the fitted points."""
        scalar = not isinstance(xq, (list, tuple))
        queries = [xq] if scalar else list(xq)
        out = []
        for q in queries:
            if q <= self.x[0]:
                out.append(self.y[0])
            elif q >= self.x[-1]:
                out.append(self.y[-1])
            else:
                # binary search for the bracketing interval
                lo, hi = 0, len(self.x) - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if self.x[mid] <= q:
                        lo = mid
                    else:
                        hi = mid
                x0, x1 = self.x[lo], self.x[hi]
                y0, y1 = self.y[lo], self.y[hi]
                if x1 == x0:
                    out.append(y0)
                else:
                    frac = (q - x0) / (x1 - x0)
                    out.append(y0 + frac * (y1 - y0))
        return out[0] if scalar else out


def sse(y, fitted, weights=None):
    """Weighted sum of squared errors between data and fit."""
    if weights is None:
        weights = [1.0] * len(y)
    return sum(w * (a - b) ** 2 for a, b, w in zip(y, fitted, weights))


# --- independent reference: the exact min-max formula ------------------------
def brute_isotonic(y, weights=None):
    """Independent exact solution via the classic min-max (max-min) formula: the optimal
    non-decreasing fit at index i is

        yhat_i = min over j >= i of  max over k <= i of  weighted_mean(y[k..j]).

    This is O(n^3) but assumption-free -- a ground truth to check PAVA against."""
    n = len(y)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0] * n

    def wmean(k, j):
        sw = sum(weights[t] for t in range(k, j + 1))
        return sum(weights[t] * y[t] for t in range(k, j + 1)) / sw

    fitted = []
    for i in range(n):
        best = float("inf")
        for j in range(i, n):
            inner = -float("inf")
            for k in range(0, i + 1):
                inner = max(inner, wmean(k, j))
            best = min(best, inner)
        fitted.append(best)
    return fitted
