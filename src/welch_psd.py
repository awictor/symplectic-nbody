"""Welch's method: a low-variance estimate of a signal's power spectral density.

The raw PERIODOGRAM -- just |DFT(x)|^2 -- is an unbiased estimate of the power spectral density (PSD),
but a terrible one: its variance does NOT shrink as the signal gets longer, so the estimate stays
jagged and noisy no matter how much data you have. Peter Welch's 1967 method fixes this by trading a
little frequency resolution for a lot of variance reduction: split the signal into overlapping SEGMENTS,
window each, compute its periodogram, and AVERAGE the periodograms. Averaging K roughly-independent
segments cuts the variance by about a factor of K, turning a hairy periodogram into a smooth, readable
spectrum -- the standard PSD estimator in every signal-analysis package.

The recipe has three knobs. The SEGMENT LENGTH sets the frequency resolution (longer = finer bins). The
OVERLAP (commonly 50%) recovers some of the data lost to windowing, giving more segments to average. The
WINDOW (Hann by default) controls spectral leakage and must be compensated for in the normalization so
the PSD integrates to the true signal power (Parseval). Welch's estimate is biased by the window's
main-lobe width but has dramatically lower variance -- the classic bias/variance trade of spectral
estimation.

This module computes the Welch PSD (and the single-segment periodogram for comparison) using the repo's
Bluestein DFT, with configurable segment length, overlap, and window. It is validated: a pure tone
produces a sharp PSD peak at its frequency; Welch's estimate has substantially lower variance than the
raw periodogram on white noise (the whole point); a white-noise PSD is approximately flat; the PSD
integrates to the signal's power (Parseval, up to window normalization); more/longer segments reduce
the variance further; and the PSD is non-negative everywhere. Reuses the repo's Bluestein DFT and the
spectrogram's window functions. Pure stdlib; the spectral-estimation companion to the spectrogram, FFT,
and Wiener-filter tools."""

from __future__ import annotations

import math

from bluestein import dft
from spectrogram import hann_window, hamming_window, rectangular_window


_WINDOWS = {"hann": hann_window, "hamming": hamming_window, "rectangular": rectangular_window}


def periodogram(x, sample_rate=1.0, window="rectangular"):
    """The raw periodogram PSD of x: |DFT(windowed x)|^2 / (fs * window_power). Returns (freqs, psd)."""
    n = len(x)
    win = _WINDOWS[window](n)
    xw = [x[i] * win[i] for i in range(n)]
    X = dft(xw)
    win_power = sum(w * w for w in win)
    scale = 1.0 / (sample_rate * win_power) if win_power > 0 else 0.0
    half = n // 2 + 1
    psd = []
    for k in range(half):
        p = (abs(X[k]) ** 2) * scale
        # fold: double the interior bins (one-sided PSD), keep DC and Nyquist single
        if 0 < k < n - k:
            p *= 2
        psd.append(p)
    freqs = [k * sample_rate / n for k in range(half)]
    return freqs, psd


def welch(x, segment_len=64, overlap=0.5, sample_rate=1.0, window="hann"):
    """Welch PSD: average the periodograms of overlapping windowed segments. Returns (freqs, psd)."""
    n = len(x)
    seg = min(segment_len, n)
    hop = max(1, int(seg * (1 - overlap)))
    win = _WINDOWS[window](seg)
    win_power = sum(w * w for w in win)
    scale = 1.0 / (sample_rate * win_power) if win_power > 0 else 0.0
    half = seg // 2 + 1
    accum = [0.0] * half
    count = 0
    i = 0
    while i + seg <= n:
        frame = x[i:i + seg]
        xw = [frame[j] * win[j] for j in range(seg)]
        X = dft(xw)
        for k in range(half):
            p = (abs(X[k]) ** 2) * scale
            if 0 < k < seg - k:
                p *= 2
            accum[k] += p
        count += 1
        i += hop
    if count == 0:
        # fall back to a single (zero-padded) segment
        return periodogram(x, sample_rate, window)
    psd = [a / count for a in accum]
    freqs = [k * sample_rate / seg for k in range(half)]
    return freqs, psd


def num_segments(n, segment_len, overlap):
    """How many overlapping segments Welch will average."""
    hop = max(1, int(segment_len * (1 - overlap)))
    count = 0
    i = 0
    while i + segment_len <= n:
        count += 1
        i += hop
    return count


def total_power(freqs, psd, sample_rate):
    """Integrate a one-sided PSD over frequency to recover the signal's average power."""
    if len(freqs) < 2:
        return psd[0] * sample_rate if psd else 0.0
    df = freqs[1] - freqs[0]
    return sum(p * df for p in psd)


def variance(vals):
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)
