"""The discrete wavelet transform -- multiresolution analysis that localises in time AND frequency.

The Fourier transform tells you WHICH frequencies a signal contains but not WHEN they occur -- a single
spike and a steady hum can have similar spectra. The discrete cosine transform improves the energy
compaction but still uses global basis functions. WAVELETS solve the localisation problem: they analyse a
signal at multiple scales simultaneously with basis functions that are compact in both time and
frequency, so you learn both what happened and where. That MULTIRESOLUTION view is why wavelets sit
behind JPEG 2000, denoising, edge detection, seismic and biomedical analysis, and the FBI's fingerprint
compression.

The mechanism is a filter bank applied recursively. At each level the signal is split by a pair of
quadrature-mirror filters into a low-pass APPROXIMATION (a coarser version of the signal) and a high-pass
DETAIL (what the coarsening threw away), each downsampled by two. Recurse on the approximation and you get
a pyramid: one small approximation plus detail coefficients at every scale, together the same size as the
input. The simplest wavelet is the HAAR (average and difference of adjacent pairs); the DAUBECHIES-4
wavelet uses a four-tap filter with two vanishing moments, so it represents smooth signals far more
sparsely -- a linear ramp compresses to almost nothing. Both are ORTHOGONAL, so the transform preserves
energy and inverts exactly by running the filter bank backwards.

This module implements the single-level and multi-level forward and inverse DWT for the Haar and
Daubechies-4 wavelets in 1D, the separable 2D transform (used on images), and a denoising helper that
thresholds small detail coefficients. Lengths are handled by periodic extension so any even length works;
powers of two decompose to the deepest level. Pure standard library -- ``math`` only.

Validation. (1) PERFECT RECONSTRUCTION: the inverse transform recovers the input to machine precision,
for single-level, multi-level, Haar and Daubechies, across many random signals and lengths. (2) ENERGY
PRESERVATION: the sum of squared coefficients equals the sum of squared inputs (Parseval), the signature
of an orthogonal transform. (3) VANISHING MOMENTS: Daubechies-4 sends a constant and a linear ramp to
(almost) zero detail coefficients -- it is blind to polynomial trends up to degree one -- while Haar is
blind only to constants. (4) SPARSITY / COMPACTION: a smooth signal's energy piles into a few large
coefficients, so keeping the top few reconstructs it at tiny error, and denoising a noisy signal by
thresholding details lowers the error toward the clean signal. (5) The 2D transform inverts exactly and a
constant image maps to a single approximation coefficient. (6) Known Haar values match by hand."""

import math


# ---------------------------------------------------------------------------
# filter banks (orthonormal coefficients)
# ---------------------------------------------------------------------------

_HAAR = [math.sqrt(0.5), math.sqrt(0.5)]

_S3 = math.sqrt(3)
_DENOM = 4 * math.sqrt(2)
_DB4 = [(1 + _S3) / _DENOM, (3 + _S3) / _DENOM, (3 - _S3) / _DENOM, (1 - _S3) / _DENOM]


def _lowpass(name):
    return _HAAR if name == "haar" else _DB4


def _highpass(lo):
    """Quadrature-mirror high-pass filter from the low-pass coefficients: g_k = (-1)^k h_{L-1-k}."""
    L = len(lo)
    return [((-1) ** k) * lo[L - 1 - k] for k in range(L)]


# ---------------------------------------------------------------------------
# single-level forward / inverse (periodic convolution + downsample)
# ---------------------------------------------------------------------------

def dwt_step(signal, wavelet="haar"):
    """One level of DWT. Returns (approximation, detail), each length len(signal)//2.

    len(signal) must be even. Uses periodic extension.
    """
    n = len(signal)
    if n % 2 != 0:
        raise ValueError("signal length must be even")
    lo = _lowpass(wavelet)
    hi = _highpass(lo)
    L = len(lo)
    half = n // 2
    approx = [0.0] * half
    detail = [0.0] * half
    for i in range(half):
        a = 0.0
        d = 0.0
        for k in range(L):
            idx = (2 * i + k) % n
            a += lo[k] * signal[idx]
            d += hi[k] * signal[idx]
        approx[i] = a
        detail[i] = d
    return approx, detail


def idwt_step(approx, detail, wavelet="haar"):
    """Invert one DWT level: reconstruct the length-2*half signal from approximation + detail."""
    half = len(approx)
    n = 2 * half
    lo = _lowpass(wavelet)
    hi = _highpass(lo)
    L = len(lo)
    out = [0.0] * n
    # transpose of the analysis: scatter each coefficient back through the filters
    for i in range(half):
        for k in range(L):
            idx = (2 * i + k) % n
            out[idx] += lo[k] * approx[i] + hi[k] * detail[i]
    return out


# ---------------------------------------------------------------------------
# multi-level transform
# ---------------------------------------------------------------------------

def dwt(signal, wavelet="haar", levels=None):
    """Multi-level DWT. Returns (approximation, [detail_1, detail_2, ...]) with detail_1 the finest.

    ``levels`` defaults to the maximum possible (down to length-1-ish while the length stays even).
    """
    a = list(signal)
    details = []
    max_levels = 0
    m = len(signal)
    while m % 2 == 0 and m >= 2:
        max_levels += 1
        m //= 2
    if levels is None:
        levels = max_levels
    levels = min(levels, max_levels)
    for _ in range(levels):
        a, d = dwt_step(a, wavelet)
        details.append(d)
    return a, details


def idwt(approx, details, wavelet="haar"):
    """Invert a multi-level DWT (details finest-first, matching dwt's output)."""
    a = list(approx)
    for d in reversed(details):
        a = idwt_step(a, d, wavelet)
    return a


def coeffs_to_flat(approx, details):
    """Pack the pyramid into one flat list [approx, deepest detail, ..., finest detail]."""
    flat = list(approx)
    for d in reversed(details):
        flat.extend(d)
    return flat


# ---------------------------------------------------------------------------
# 2D separable transform
# ---------------------------------------------------------------------------

def dwt2_step(image, wavelet="haar"):
    """One level of 2D DWT. Returns (LL, LH, HL, HH) sub-bands."""
    # transform rows
    rows_a, rows_d = [], []
    for row in image:
        a, d = dwt_step(row, wavelet)
        rows_a.append(a)
        rows_d.append(d)
    # transform columns of each half
    def cols(mat):
        t = _T(mat)
        ca, cd = [], []
        for col in t:
            a, d = dwt_step(col, wavelet)
            ca.append(a)
            cd.append(d)
        return _T(ca), _T(cd)
    LL, HL = cols(rows_a)
    LH, HH = cols(rows_d)
    return LL, LH, HL, HH


def idwt2_step(LL, LH, HL, HH, wavelet="haar"):
    """Invert one level of the 2D DWT."""
    def icols(a_cols, d_cols):
        ta, td = _T(a_cols), _T(d_cols)
        out = [idwt_step(ta[i], td[i], wavelet) for i in range(len(ta))]
        return _T(out)
    rows_a = icols(LL, HL)
    rows_d = icols(LH, HH)
    return [idwt_step(rows_a[i], rows_d[i], wavelet) for i in range(len(rows_a))]


def _T(m):
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


# ---------------------------------------------------------------------------
# denoising
# ---------------------------------------------------------------------------

def threshold_details(details, threshold):
    """Soft-threshold every detail coefficient (shrink toward zero), returning new detail lists."""
    out = []
    for d in details:
        nd = []
        for c in d:
            if c > threshold:
                nd.append(c - threshold)
            elif c < -threshold:
                nd.append(c + threshold)
            else:
                nd.append(0.0)
        out.append(nd)
    return out


def denoise(signal, wavelet="haar", threshold=0.5, levels=None):
    """Wavelet denoising: transform, soft-threshold the details, invert."""
    approx, details = dwt(signal, wavelet, levels)
    return idwt(approx, threshold_details(details, threshold), wavelet)


def energy(seq):
    return sum(v * v for v in seq)
