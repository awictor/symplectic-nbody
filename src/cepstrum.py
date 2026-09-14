"""Cepstrum: the spectrum of a log-spectrum, for pitch tracking and echo detection.

The CEPSTRUM (an anagram of "spectrum") is what you get by taking the Fourier transform of the LOG
magnitude spectrum of a signal. Bogert, Healy, and Tukey coined it in 1963 -- along with the deliberate
wordplay QUEFRENCY (an anagram of "frequency") for its independent axis and LIFTERING for filtering in
that domain -- while studying seismic echoes. The trick is that many signals are a slowly-varying
ENVELOPE convolved with a rapidly-varying EXCITATION (voiced speech = vocal-tract filter * glottal
pulse train; a signal with an echo = original * a two-spike impulse response). Convolution multiplies
their spectra, and the LOGARITHM turns that product into a SUM -- so in the cepstral domain the envelope
and the excitation land in DIFFERENT quefrency ranges and can be read off or separated linearly.

Two classic payoffs. PITCH DETECTION: a periodic signal of period T has harmonics spaced 1/T apart in
the spectrum, which is itself a periodic ripple, so the cepstrum shows a sharp peak at quefrency T --
robustly recovering the fundamental even when it is weak or missing. ECHO DETECTION: a signal plus a
delayed copy has a spectrum modulated by a cosine of period 1/delay, giving a cepstral peak at the echo
delay. This module computes the real and complex (and power) cepstrum via the repo's arbitrary-length
DFT, and extracts the pitch/echo quefrency peak. It is validated: a pure impulse train of period T
produces its dominant cepstral peak at quefrency T; a synthetic voiced-speech-like signal recovers its
fundamental period; a signal with an echo at delay d shows a cepstral peak at d; the cepstrum of white
noise has no dominant low-quefrency peak; and the real cepstrum is real-valued and symmetric for a
real input. Reuses the repo's Bluestein DFT. Pure stdlib; the spectral-analysis companion to the FFT,
chirp-Z, and Goertzel tools."""

from __future__ import annotations

import cmath
import math

from bluestein import dft, idft


def real_cepstrum(x):
    """Real cepstrum: IDFT( log|DFT(x)| ). Returns a real-valued sequence."""
    X = dft(x)
    logmag = [math.log(abs(v) + 1e-12) for v in X]
    c = idft(logmag)
    return [v.real for v in c]


def power_cepstrum(x):
    """Power cepstrum: | IDFT( log|DFT(x)|^2 ) |^2."""
    X = dft(x)
    logp = [math.log(abs(v) ** 2 + 1e-24) for v in X]
    c = idft(logp)
    return [abs(v) ** 2 for v in c]


def complex_cepstrum(x):
    """Complex cepstrum: IDFT( log(DFT(x)) ) with the complex logarithm (unwrapped phase)."""
    X = dft(x)
    logX = []
    prev_phase = 0.0
    offset = 0.0
    for v in X:
        mag = abs(v) + 1e-12
        phase = cmath.phase(v)
        # simple phase unwrapping
        d = phase - prev_phase
        if d > math.pi:
            offset -= 2 * math.pi
        elif d < -math.pi:
            offset += 2 * math.pi
        prev_phase = phase
        logX.append(complex(math.log(mag), phase + offset))
    c = idft(logX)
    return c


def pitch_quefrency(x, min_q=2, max_q=None):
    """Estimate the pitch period (in samples) as the quefrency of the dominant cepstral peak.

    Searches quefrencies in [min_q, max_q] (default up to n//2) to skip the low-quefrency envelope."""
    c = real_cepstrum(x)
    n = len(c)
    if max_q is None:
        max_q = n // 2
    max_q = min(max_q, n - 1)
    best_q = min_q
    best_v = -math.inf
    for q in range(min_q, max_q + 1):
        if c[q] > best_v:
            best_v = c[q]
            best_q = q
    return best_q


def echo_delay(x, min_q=2, max_q=None):
    """Estimate an echo delay (in samples) from the dominant cepstral peak (same as pitch_quefrency)."""
    return pitch_quefrency(x, min_q, max_q)


def pitch_hz(x, sample_rate, min_hz=50.0, max_hz=500.0):
    """Estimate fundamental frequency in Hz via the cepstral peak within [min_hz, max_hz]."""
    n = len(x)
    min_q = max(2, int(sample_rate / max_hz))
    max_q = min(n - 1, int(sample_rate / min_hz))
    q = pitch_quefrency(x, min_q, max_q)
    return sample_rate / q if q > 0 else 0.0
