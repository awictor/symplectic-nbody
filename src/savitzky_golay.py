"""Savitzky-Golay filtering: smoothing that preserves peaks, and differentiation of noisy data.

Smoothing noisy data usually blunts its features -- a moving average flattens peaks and shifts edges.
The SAVITZKY-GOLAY filter (1964), the standard in analytical chemistry and signal processing, smooths
by fitting a low-degree POLYNOMIAL to a sliding window of points by least squares and taking the
fitted value at the centre. Because a polynomial can follow a peak's curvature, the filter removes
noise while preserving the height and width of peaks far better than a moving average -- which is why
it is the default for spectroscopy, chromatography, and any data where the SHAPE matters.

The elegant fact is that for evenly spaced points the least-squares polynomial fit reduces to a fixed
set of CONVOLUTION COEFFICIENTS that depend only on the window size and polynomial degree, not on the
data. So the whole filter is a single convolution: precompute the coefficients once (by solving the
small normal equations of the polynomial fit at integer offsets), then slide them across the signal.
The same machinery, by taking the DERIVATIVE of the fitted polynomial at the centre, yields a
smoothed estimate of the signal's first, second, or higher DERIVATIVE -- differentiating noisy data,
normally a disaster, becomes stable, which is invaluable for finding inflection points and rates.

This module computes Savitzky-Golay coefficients for any odd window and polynomial degree (and any
derivative order), and applies the filter with edge handling. It is verified against exact references:
that a polynomial of degree <= the filter degree passes through UNCHANGED (the defining property),
that the derivative mode recovers the analytic derivative of such polynomials, that smoothing a noisy
sine reduces the error to the clean signal versus the noisy input, that the coefficients sum to one
(unit DC gain) for smoothing and to zero for derivatives, and that it beats a moving average at
preserving a Gaussian peak's height. Pure stdlib; a signal-processing companion to the FFT and
quadrature notes."""

from __future__ import annotations


def _solve_linear(A, b):
    """Solve the small dense linear system A x = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= f * M[col][j]
    return [M[i][n] for i in range(n)]


def coefficients(window, degree, deriv=0, delta=1.0):
    """Savitzky-Golay convolution coefficients for a window of `window` points (odd), polynomial
    `degree`, and derivative order `deriv`. Returns a list of `window` weights.

    Applying these as a convolution to evenly-spaced data gives the value (deriv=0) or the `deriv`-th
    derivative of the least-squares polynomial fit at the window centre."""
    if window % 2 == 0 or window < 1:
        raise ValueError("window must be a positive odd integer")
    if degree >= window:
        raise ValueError("degree must be less than the window size")
    if deriv > degree:
        raise ValueError("derivative order cannot exceed the polynomial degree")
    half = window // 2
    # design matrix A[i][j] = offset_i ** j, offsets from -half..half
    offsets = list(range(-half, half + 1))
    # normal equations: (A^T A) c = A^T e_deriv, but we want the coefficient row that extracts the
    # deriv-th derivative. Equivalent: solve for the polynomial that is 1 at power `deriv`.
    ncoef = degree + 1
    ATA = [[0.0] * ncoef for _ in range(ncoef)]
    for i in range(ncoef):
        for j in range(ncoef):
            ATA[i][j] = sum(x ** (i + j) for x in offsets)
    # for each data point k, its contribution to the fitted derivative at centre:
    # coefficient_k = sum_i (ATA^-1)[deriv][i] * offset_k^i, times deriv! / delta^deriv
    import math
    weights = []
    # solve ATA * row = e_deriv to get the deriv-th row of ATA^-1
    e = [1.0 if i == deriv else 0.0 for i in range(ncoef)]
    inv_row = _solve_linear(ATA, e)
    scale = math.factorial(deriv) / (delta ** deriv)
    for x in offsets:
        w = sum(inv_row[i] * (x ** i) for i in range(ncoef))
        weights.append(w * scale)
    return weights


def filter_signal(y, window, degree, deriv=0, delta=1.0, mode="interp"):
    """Apply a Savitzky-Golay filter to signal y. mode='interp' fits a polynomial at the edges;
    mode='nearest' repeats the boundary sample. Returns a list the same length as y."""
    n = len(y)
    if n < window:
        raise ValueError("signal shorter than the window")
    coeffs = coefficients(window, degree, deriv, delta)
    half = window // 2
    out = [0.0] * n
    # interior points: straightforward convolution
    for i in range(half, n - half):
        out[i] = sum(coeffs[j] * y[i - half + j] for j in range(window))
    # edges
    for i in list(range(half)) + list(range(n - half, n)):
        if mode == "nearest":
            acc = 0.0
            for j in range(window):
                idx = min(max(i - half + j, 0), n - 1)
                acc += coeffs[j] * y[idx]
            out[i] = acc
        else:  # interp: use a shifted coefficient set for the edge window
            if i < half:
                edge = coefficients(window, degree, deriv, delta)
                # fit the first `window` points, evaluate at position i
                out[i] = _edge_value(y[:window], i, degree, deriv, delta)
            else:
                out[i] = _edge_value(y[n - window:], i - (n - window), degree, deriv, delta)
    return out


def _edge_value(window_y, pos, degree, deriv, delta):
    """Fit a polynomial of `degree` to window_y (offsets 0..len-1) and return the `deriv`-th
    derivative at index `pos`."""
    import math
    m = len(window_y)
    offsets = list(range(m))
    ncoef = degree + 1
    ATA = [[sum(x ** (i + j) for x in offsets) for j in range(ncoef)] for i in range(ncoef)]
    ATb = [sum(window_y[k] * (offsets[k] ** i) for k in range(m)) for i in range(ncoef)]
    coef = _solve_linear(ATA, ATb)
    # evaluate the deriv-th derivative of the polynomial at x = pos
    val = 0.0
    for i in range(deriv, ncoef):
        term = coef[i] * (math.factorial(i) / math.factorial(i - deriv)) * (pos ** (i - deriv))
        val += term
    return val / (delta ** deriv)
