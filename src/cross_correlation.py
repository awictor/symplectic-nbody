"""Cross-correlation and the matched filter: finding a signal, and where it hides, inside noise.

CROSS-CORRELATION slides one signal past another and, at each lag, measures how much they overlap:

    (x corr y)[k] = sum_n x[n] y[n - k].

Its peak tells you the LAG at which the two signals line up best -- the basis of radar and sonar
ranging (how long until the echo returns), GPS code acquisition, audio time-delay estimation, and
template matching. AUTOCORRELATION is the special case x corr x, whose peak sits at lag 0 (a signal is
always most similar to itself, unshifted) and whose secondary peaks reveal PERIODICITY -- it is how
pitch detection and seasonality analysis work.

The MATCHED FILTER is the optimal detector for a known signal buried in white noise: correlate the
noisy input against a time-reversed copy of the template. Among all linear filters it MAXIMIZES the
signal-to-noise ratio at the moment of detection (a result from detection theory), which is why every
radar, pulse-compression, and communications receiver is built around one. Cross-correlation is
matched filtering, and both are just convolution with a reversed kernel -- so the repo's FFT
convolution computes them in O(n log n).

This module implements cross-correlation and autocorrelation (direct and FFT-based), lag estimation,
normalized correlation (the correlation coefficient as a function of lag), and a matched-filter
detector. It is validated exactly: the autocorrelation peaks at lag 0 and equals the signal energy
there; cross-correlating a signal with a delayed copy of itself puts the peak at exactly the delay;
autocorrelation of a periodic signal peaks at multiples of its period; the FFT and direct methods
agree; the normalized correlation of a signal with itself is 1 at lag 0; and the matched filter
recovers the location of a known pulse hidden in noise far better than a raw threshold. Pure stdlib;
the detection-and-alignment companion to the FFT, Goertzel, and convolution tools."""

from __future__ import annotations

from fft import convolve


def cross_correlation(x, y):
    """Full cross-correlation of x and y. Returns (lags, values) with lag = index shift of y.

    values[k] corresponds to lag = k - (len(y) - 1); a peak at positive lag means y is DELAYED
    relative to x (y must slide left to align). Uses FFT convolution with y reversed.
    """
    yr = list(reversed(y))
    conv = convolve(list(x), yr)   # length len(x)+len(y)-1
    lags = [k - (len(y) - 1) for k in range(len(conv))]
    return lags, conv


def autocorrelation(x):
    """Autocorrelation of x: cross-correlation with itself. Peak at lag 0."""
    return cross_correlation(x, x)


def correlation_direct(x, y):
    """Direct O(n*m) cross-correlation (reference), same convention as cross_correlation."""
    n, m = len(x), len(y)
    lags = list(range(-(m - 1), n))
    values = []
    for lag in lags:
        s = 0.0
        for j in range(m):
            i = lag + j
            if 0 <= i < n:
                s += x[i] * y[j]
        values.append(s)
    return lags, values


def best_lag(x, y):
    """The lag at which y best aligns with x (index of the cross-correlation peak)."""
    lags, values = cross_correlation(x, y)
    best = max(range(len(values)), key=lambda k: values[k])
    return lags[best]


def estimate_delay(reference, delayed):
    """Estimate the integer delay of `delayed` relative to `reference` by the correlation peak."""
    return best_lag(delayed, reference)


def normalized_correlation(x, y):
    """Normalized cross-correlation (correlation coefficient in [-1, 1]) as a function of lag."""
    import math
    lags, raw = cross_correlation(x, y)
    ex = math.sqrt(sum(v * v for v in x))
    ey = math.sqrt(sum(v * v for v in y))
    denom = ex * ey
    if denom == 0:
        return lags, [0.0] * len(raw)
    return lags, [v / denom for v in raw]


def matched_filter(signal, template):
    """Matched-filter detection: correlate signal against the template, return (lags, response).

    The response peaks at the lag where the template best matches -- the maximum-SNR detector for a
    known template in white noise.
    """
    return cross_correlation(signal, template)


def detect_pulse(signal, template):
    """Return the index in `signal` where `template` most likely starts (matched-filter argmax)."""
    lags, resp = matched_filter(signal, template)
    best = max(range(len(resp)), key=lambda k: resp[k])
    return lags[best]


def signal_energy(x):
    """Sum of squares of x (the autocorrelation value at lag 0)."""
    return sum(v * v for v in x)
