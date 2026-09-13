"""Hilbert transform and the analytic signal: amplitude and instantaneous frequency of a wave.

A real signal x(t) hides two things you often want: its ENVELOPE (the slowly-varying amplitude, like
the outline of an AM radio wave) and its INSTANTANEOUS FREQUENCY (how fast the phase is turning right
now, which changes over time in a chirp or vibrato). Both come from the ANALYTIC SIGNAL -- a complex
signal whose real part is x(t) and whose imaginary part is the HILBERT TRANSFORM of x, a 90-degree
phase shift of every frequency component. The analytic signal z(t) = x(t) + i*H[x](t) = A(t) e^{i
phi(t)} then hands you the envelope A(t) = |z(t)| and the phase phi(t); the derivative of the phase is
the instantaneous frequency.

The clean way to build it is in the frequency domain. Take the DFT of x, ZERO OUT the negative
frequencies, DOUBLE the positive ones (leave DC and Nyquist alone), and inverse-transform: the result
is exactly the analytic signal. The Hilbert transform is its imaginary part. This is the standard
method behind envelope detection in communications, the amplitude/frequency demodulation of AM/FM,
the empirical-mode-decomposition analysis of nonstationary signals, and vibration diagnostics.

This module computes the Hilbert transform, the analytic signal, the amplitude envelope, and the
instantaneous phase (unwrapped) and frequency, using a self-contained O(n^2) DFT so any signal length
works. Validated against theory: the Hilbert transform of cos is sin (and of sin is -cos), a 90-degree
shift; the envelope of an amplitude-modulated carrier recovers the modulating amplitude; the
instantaneous frequency of a linear chirp rises linearly at the right rate; the transform is linear;
and applying it twice negates the signal (H[H[x]] = -x). Pure stdlib; the demodulation companion to
the FFT and the Goertzel / spectral tools."""

from __future__ import annotations

import cmath
import math


def _dft(x):
    n = len(x)
    return [sum(x[j] * cmath.exp(-2j * math.pi * j * k / n) for j in range(n)) for k in range(n)]


def _idft(X):
    n = len(X)
    return [sum(X[k] * cmath.exp(2j * math.pi * j * k / n) for k in range(n)) / n for j in range(n)]


def analytic_signal(x):
    """The analytic signal of a real sequence x: complex z with Re(z) = x and Im(z) = Hilbert(x).
    Built by zeroing negative frequencies and doubling positive ones in the DFT."""
    n = len(x)
    X = _dft([complex(v) for v in x])
    h = [0.0] * n
    if n % 2 == 0:
        h[0] = 1.0            # DC
        h[n // 2] = 1.0       # Nyquist
        for k in range(1, n // 2):
            h[k] = 2.0        # positive frequencies doubled
        # negative frequencies (n//2+1 .. n-1) stay 0
    else:
        h[0] = 1.0
        for k in range(1, (n + 1) // 2):
            h[k] = 2.0
    Z = [X[k] * h[k] for k in range(n)]
    return _idft(Z)


def hilbert_transform(x):
    """The Hilbert transform of x: the imaginary part of its analytic signal (a 90-degree phase
    shift of every frequency component)."""
    z = analytic_signal(x)
    return [v.imag for v in z]


def envelope(x):
    """The amplitude envelope A(t) = |analytic signal|."""
    return [abs(v) for v in analytic_signal(x)]


def instantaneous_phase(x, unwrap=True):
    """The instantaneous phase phi(t) = arg(analytic signal), optionally unwrapped to be continuous."""
    z = analytic_signal(x)
    phase = [math.atan2(v.imag, v.real) for v in z]
    if unwrap:
        return _unwrap(phase)
    return phase


def _unwrap(phase):
    """Remove 2-pi jumps so the phase is continuous."""
    out = [phase[0]]
    offset = 0.0
    for i in range(1, len(phase)):
        d = phase[i] - phase[i - 1]
        if d > math.pi:
            offset -= 2 * math.pi
        elif d < -math.pi:
            offset += 2 * math.pi
        out.append(phase[i] + offset)
    return out


def instantaneous_frequency(x, sample_rate=1.0):
    """Instantaneous frequency (cycles per unit time) = derivative of the unwrapped phase / 2pi,
    times the sample rate. Returns n-1 central-ish differences."""
    phase = instantaneous_phase(x, unwrap=True)
    n = len(phase)
    freq = []
    for i in range(n - 1):
        dphi = phase[i + 1] - phase[i]
        freq.append(dphi / (2 * math.pi) * sample_rate)
    return freq
