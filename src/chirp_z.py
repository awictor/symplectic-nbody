"""Chirp Z-transform: sample the z-transform along any spiral, for zoom-FFT spectral analysis.

The DFT evaluates a signal's z-transform at N points equally spaced on the unit circle -- fixed count,
fixed spacing, whole circle. The CHIRP Z-TRANSFORM (Rabiner, Schafer, Rader 1969) frees all three
knobs: it evaluates the z-transform at M points along an arbitrary logarithmic SPIRAL in the complex
plane, z_k = A * W^{-k}, where A sets the starting point and W the ratio between successive points. Pick
A and W on the unit circle over a narrow angular range and you get ZOOM-FFT: a fine-resolution DFT of
just a slice of the spectrum, resolving closely-spaced tones that a same-length ordinary FFT would smear
into one bin. Pick a spiral that leaves the circle and you probe damped/growing modes -- the reason CZT
appears in modal analysis and filter design.

The trick that makes it fast is Bluestein's identity: the product n*k in the exponent is rewritten via
n*k = (n^2 + k^2 - (k-n)^2)/2, turning the transform into a chirp premultiply, a CONVOLUTION against a
chirp kernel, and a chirp postmultiply. The convolution is done by FFT, so the whole CZT costs
O((N+M) log(N+M)) regardless of how few or many output points you ask for, and independent of their
spacing.

This module implements the general chirp Z-transform for arbitrary A, W, and output count M, plus a
zoom-FFT helper that maps a frequency band to the right spiral parameters, reusing the repo's Bluestein
convolution. It is validated: with A=1, W=exp(-2πi/N), and M=N the CZT reduces exactly to the ordinary
DFT; a zoom over the full band reproduces the DFT bins; the CZT of a pure tone peaks at the correct
frequency; zoom-FFT resolves two tones closer together than the FFT bin width (where a same-length FFT
cannot); a constant signal transforms to the expected geometric response; and linearity holds. Reuses
the repo's Bluestein convolution. Pure stdlib; the spectral-analysis companion to the FFT, Bluestein,
and Goertzel tools."""

from __future__ import annotations

import cmath
import math

from bluestein import convolve as _bluestein_convolve


def czt(x, m=None, w=None, a=1.0):
    """Chirp Z-transform of x at m points z_k = a * w^{-k}, k = 0..m-1.

    x: input sequence (real or complex). m: number of output points (default len(x)).
    w: ratio between points (default exp(-2πi/m), i.e. points on the unit circle). a: starting point."""
    n = len(x)
    if m is None:
        m = n
    if w is None:
        w = cmath.exp(-2j * cmath.pi / m)
    a = complex(a)
    w = complex(w)
    # chirp premultiply: y[k] = x[k] * a^{-k} * w^{k^2/2}
    y = [x[k] * (a ** (-k)) * (w ** (k * k / 2.0)) for k in range(n)]
    # chirp kernel v[k] = w^{-k^2/2}, for k in -(n-1)..(m-1)
    L = n + m - 1
    v = [w ** (-((k - (n - 1)) ** 2) / 2.0) for k in range(L)]
    # linear convolution of y (length n) with v (length L)
    conv = _bluestein_convolve(y, v)
    # the desired outputs are conv[n-1 .. n-1+m-1], times the chirp postmultiply w^{k^2/2}
    out = []
    for k in range(m):
        val = conv[k + n - 1] * (w ** (k * k / 2.0))
        out.append(val)
    return out


def zoom_fft(x, f_lo, f_hi, m, sample_rate=1.0):
    """Zoom-FFT: evaluate the spectrum of x at m equally-spaced frequencies in [f_lo, f_hi].

    Returns (freqs, spectrum). Frequencies in the same units as sample_rate."""
    # map frequency band to spiral parameters: start angle theta0, step dtheta on the unit circle
    theta0 = 2 * math.pi * f_lo / sample_rate
    dtheta = 2 * math.pi * (f_hi - f_lo) / (sample_rate * (m - 1)) if m > 1 else 0.0
    a = cmath.exp(1j * theta0)                    # z_0 = e^{i theta0}
    w = cmath.exp(-1j * dtheta)                   # z_k = a * w^{-k} = e^{i(theta0 + k dtheta)}
    spectrum = czt(x, m=m, w=w, a=a)
    freqs = [f_lo + (f_hi - f_lo) * k / (m - 1) if m > 1 else f_lo for k in range(m)]
    return freqs, spectrum


def czt_as_dft(x):
    """CZT specialized to the ordinary DFT (A=1, W=exp(-2πi/N), M=N) -- for validation."""
    n = len(x)
    return czt(x, m=n, w=cmath.exp(-2j * cmath.pi / n), a=1.0)


def magnitude(spectrum):
    return [abs(c) for c in spectrum]
