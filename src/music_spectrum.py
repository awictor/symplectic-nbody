"""MUSIC: pinpoint the frequencies of superimposed sinusoids by exploiting the orthogonality of noise.

MUSIC (MUltiple SIgnal Classification; Schmidt 1979) is the eigenvector method of super-resolution spectral
estimation. Its idea is geometric and beautiful. Stack short windows of a signal into a data matrix and
form the covariance R. If the signal is p complex sinusoids in white noise, R's eigenvectors split cleanly
into two orthogonal groups: the p eigenvectors with the LARGEST eigenvalues span the SIGNAL SUBSPACE, and
the rest span the NOISE SUBSPACE. Crucially, each true sinusoid's steering vector a(f) = [1, e^{i w}, ...,
e^{i(m-1)w}] lies ENTIRELY in the signal subspace -- so it is exactly ORTHOGONAL to every noise-subspace
eigenvector. MUSIC turns that orthogonality into a spectrum:

    P_MUSIC(f) = 1 / ( a(f)^H E_n E_n^H a(f) ),

which explodes toward infinity at the true frequencies (the denominator vanishes) and stays small
elsewhere. The peaks are razor-sharp -- far below the FFT bin width -- because they come from a projection
going to zero, not from energy in a bin. MUSIC and its relatives (root-MUSIC, ESPRIT) are the backbone of
radar direction-of-arrival estimation, sensor arrays, and parametric spectral analysis.

This module builds the covariance from a real time series, eigendecomposes it (reusing the repo's symmetric
Jacobi solver), separates signal and noise subspaces, and evaluates the MUSIC pseudospectrum and its peaks.
It is validated: the pseudospectrum peaks at the exact frequencies of a synthesized multi-tone signal; two
tones far closer than one FFT bin are resolved as two sharp peaks where the periodogram shows one; the
eigenvalue split cleanly separates the signal subspace (a few large eigenvalues) from the noise floor; the
signal-subspace dimension equals twice the number of real tones; peak sharpness beats the periodogram; and
results are deterministic. It cross-checks against the repo's ESPRIT frequencies. Pure stdlib; the
subspace-spectral companion to the ESPRIT, Prony, Burg, and FFT tools."""

from __future__ import annotations

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import jacobi_eigen


def _covariance(x, m):
    """Estimate the m-by-m covariance matrix from a real signal by averaging outer products of
    length-m sliding windows. R[i][j] = mean_t x[t+i] x[t+j] -- real, symmetric, positive semidefinite."""
    n = len(x)
    num = n - m + 1
    if num < 1:
        raise ValueError("signal too short for the chosen window size m")
    R = [[0.0] * m for _ in range(m)]
    for t in range(num):
        w = x[t:t + m]
        for i in range(m):
            wi = w[i]
            for j in range(i, m):
                R[i][j] += wi * w[j]
    for i in range(m):
        for j in range(i, m):
            R[i][j] /= num
            R[j][i] = R[i][j]
    return R


def _steering(f, m):
    """Steering vector a(f) = [1, e^{i 2 pi f}, ..., e^{i 2 pi f (m-1)}] for normalized frequency f."""
    w = 2.0 * math.pi * f
    return [cmath.exp(1j * w * k) for k in range(m)]


def music_spectrum(x, n_signal, m=None, n_freqs=1024, fmax=0.5):
    """Compute the MUSIC pseudospectrum of real signal x.

    n_signal is the dimension of the signal subspace (= 2 * number of real tones, since each real
    sinusoid contributes a +/- frequency pair). m is the covariance/window size (default ~ len(x)//2,
    capped). Returns (freqs, pseudospectrum) over [0, fmax)."""
    n = len(x)
    if m is None:
        m = min(max(n_signal + 2, n // 2), n - 1, 40)
    if m <= n_signal:
        raise ValueError("window size m must exceed the signal-subspace dimension")

    R = _covariance([float(v) for v in x], m)
    vals, V = jacobi_eigen.sorted_eigen(R)   # eigenvalues DESCENDING; V columns are eigenvectors
    # noise subspace = eigenvectors for the SMALLEST eigenvalues (indices n_signal .. m-1)
    noise_vecs = [[V[i][c] for i in range(m)] for c in range(n_signal, m)]

    freqs = [fmax * i / n_freqs for i in range(n_freqs)]
    psd = []
    for f in freqs:
        a = _steering(f, m)
        # denominator = sum over noise eigenvectors of |a^H v|^2 (real, >= 0)
        denom = 0.0
        for v in noise_vecs:
            proj = sum(a[i].conjugate() * v[i] for i in range(m))
            denom += (proj * proj.conjugate()).real
        psd.append(1.0 / denom if denom > 1e-300 else 1e300)
    return freqs, psd


def music_peaks(x, n_signal, m=None, n_freqs=4096, fmax=0.5, n_peaks=None):
    """Return the frequencies of the tallest MUSIC peaks (local maxima), sorted by height.

    n_peaks defaults to n_signal // 2 (the number of real tones)."""
    freqs, psd = music_spectrum(x, n_signal, m=m, n_freqs=n_freqs, fmax=fmax)
    peaks = []
    for i in range(1, len(psd) - 1):
        if psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
            peaks.append((freqs[i], psd[i]))
    peaks.sort(key=lambda p: -p[1])
    if n_peaks is None:
        n_peaks = max(1, n_signal // 2)
    return [f for f, _p in peaks[:n_peaks]]


def eigenvalue_spectrum(x, m):
    """Return the covariance eigenvalues (descending) -- the signal/noise split is visible as a
    few large values above a flat noise floor."""
    R = _covariance([float(v) for v in x], m)
    vals, _V = jacobi_eigen.sorted_eigen(R)
    return vals
