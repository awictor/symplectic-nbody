"""Lomb-Scargle periodogram: finding periods in unevenly-sampled data, where the FFT can't go.

The FFT needs samples on a regular grid. Real time series often aren't: a variable star is observed
only on clear nights, a radial-velocity planet hunt gets telescope time in scattered blocks, a
patient's measurements come at irregular intervals. Interpolating to a grid distorts the spectrum.
The LOMB-SCARGLE periodogram (Lomb 1976, Scargle 1982) sidesteps this: it fits a sinusoid of each
trial frequency by LEAST SQUARES directly to the irregular samples and reports how much variance that
sinusoid explains. Peaks in the resulting power spectrum mark real periodicities -- and Scargle chose
the formula's time-offset so the statistic has the same simple chi-square distribution it would for
evenly-spaced data, which is what makes peak SIGNIFICANCE computable.

For each angular frequency omega, a time offset tau is chosen so the sine and cosine basis are
orthogonal over the sample times:

    tan(2 omega tau) = (sum sin 2 omega t) / (sum cos 2 omega t),

and the power is

    P(omega) = 1/2 [ (sum y cos omega(t-tau))^2 / sum cos^2 omega(t-tau)
                   + (sum y sin omega(t-tau))^2 / sum sin^2 omega(t-tau) ].

This module computes the classical periodogram (mean-subtracted), the normalized version whose peak
height gives a false-alarm probability under the null hypothesis of Gaussian noise, the best-fit
period, and a helper to build a sensible frequency grid from the data's time span and sampling. It
also reconstructs the best-fit sinusoid's amplitude and phase at any frequency.

Validated against ground truth and the FFT: on data sampled from a known sinusoid (even OR randomly
uneven) the periodogram peaks at the injected frequency; on evenly-spaced data the peak frequency and
relative heights match a direct FFT power spectrum; adding a second tone produces two peaks; the
recovered amplitude matches the injected one; and the false-alarm probability is near 1 for pure
noise and near 0 at a strong real peak. Pure stdlib; the uneven-sampling companion to the FFT and the
autocorrelation-based spectral tools."""

from __future__ import annotations

import math


def frequency_grid(times, samples_per_peak=5, nyquist_factor=1.0):
    """A reasonable trial-frequency grid: from 1/baseline up to a pseudo-Nyquist based on the
    median sample spacing. Returns a list of ordinary (not angular) frequencies, excluding zero."""
    t = sorted(times)
    n = len(t)
    if n < 2:
        raise ValueError("need at least two time points")
    baseline = t[-1] - t[0]
    if baseline <= 0:
        raise ValueError("degenerate time span")
    # median spacing -> pseudo-Nyquist
    diffs = sorted(t[i + 1] - t[i] for i in range(n - 1))
    med = diffs[len(diffs) // 2]
    f_nyq = 0.5 / med * nyquist_factor
    df = 1.0 / (samples_per_peak * baseline)
    freqs = []
    f = df
    while f <= f_nyq:
        freqs.append(f)
        f += df
    return freqs


def periodogram(times, values, freqs, normalize=True):
    """Lomb-Scargle power at each frequency in `freqs` (ordinary frequencies, cycles per unit time).
    If normalize, powers are scaled by the data variance so the false-alarm formula applies."""
    n = len(values)
    if n != len(times):
        raise ValueError("times and values must have equal length")
    if n < 2:
        raise ValueError("need at least two samples")
    mean = sum(values) / n
    y = [v - mean for v in values]
    var = sum(yi * yi for yi in y) / n
    powers = []
    for f in freqs:
        w = 2 * math.pi * f
        # tau so the basis is orthogonal
        s2 = sum(math.sin(2 * w * t) for t in times)
        c2 = sum(math.cos(2 * w * t) for t in times)
        tau = math.atan2(s2, c2) / (2 * w) if w != 0 else 0.0
        yc = ys = cc = ss = 0.0
        for t, yi in zip(times, y):
            arg = w * (t - tau)
            cos_a = math.cos(arg)
            sin_a = math.sin(arg)
            yc += yi * cos_a
            ys += yi * sin_a
            cc += cos_a * cos_a
            ss += sin_a * sin_a
        p = 0.0
        if cc > 1e-300:
            p += yc * yc / cc
        if ss > 1e-300:
            p += ys * ys / ss
        p *= 0.5
        if normalize and var > 0:
            p /= var
        powers.append(p)
    return powers


def best_frequency(times, values, freqs=None, **grid_kwargs):
    """The frequency of maximum Lomb-Scargle power. Builds a grid if freqs is None."""
    if freqs is None:
        freqs = frequency_grid(times, **grid_kwargs)
    powers = periodogram(times, values, freqs)
    best = max(range(len(freqs)), key=lambda i: powers[i])
    return freqs[best], powers[best]


def best_period(times, values, freqs=None, **grid_kwargs):
    """The dominant period (1 / best frequency)."""
    f, _ = best_frequency(times, values, freqs, **grid_kwargs)
    return 1.0 / f if f > 0 else float("inf")


def false_alarm_probability(power, n_frequencies):
    """Probability that a normalized-periodogram peak this high arises from Gaussian noise alone.
    For the standard normalization the single-frequency exceedance probability is exp(-power);
    with M independent trial frequencies the peak FAP is 1 - (1 - exp(-power))^M."""
    single = math.exp(-power)
    return 1.0 - (1.0 - single) ** n_frequencies


def fit_sinusoid(times, values, freq):
    """Least-squares best-fit y ~ mean + A cos(2 pi f t + phi) at a fixed frequency.
    Returns (amplitude, phase, offset)."""
    n = len(values)
    w = 2 * math.pi * freq
    # design matrix columns: cos, sin, 1  -> solve 3x3 normal equations
    Scc = Sss = Scs = Sc = Ss = S1 = 0.0
    Syc = Sys = Sy = 0.0
    for t, y in zip(times, values):
        c = math.cos(w * t)
        s = math.sin(w * t)
        Scc += c * c
        Sss += s * s
        Scs += c * s
        Sc += c
        Ss += s
        S1 += 1
        Syc += y * c
        Sys += y * s
        Sy += y
    # normal equations M [a b d]^T = r, where model = a cos + b sin + d
    M = [[Scc, Scs, Sc], [Scs, Sss, Ss], [Sc, Ss, S1]]
    r = [Syc, Sys, Sy]
    a, b, d = _solve3(M, r)
    amp = math.hypot(a, b)
    phase = math.atan2(-b, a)
    return amp, phase, d


def _solve3(M, r):
    """Solve a 3x3 linear system by Cramer's rule."""
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det3(M)
    if abs(D) < 1e-300:
        raise ValueError("singular system")
    out = []
    for col in range(3):
        Mc = [row[:] for row in M]
        for row in range(3):
            Mc[row][col] = r[row]
        out.append(det3(Mc) / D)
    return out
