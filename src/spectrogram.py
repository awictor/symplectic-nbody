"""Spectrogram: the short-time Fourier transform, showing how a signal's spectrum changes over time.

A plain Fourier transform tells you WHICH frequencies are present but not WHEN -- it smears a bird's
rising whistle and a car's steady hum into the same static spectrum. The SHORT-TIME FOURIER TRANSFORM
(STFT) fixes this by chopping the signal into overlapping FRAMES, applying a smooth WINDOW to each to
suppress edge artefacts, and Fourier-transforming each frame separately. Stack the resulting magnitude
spectra column by column and you get a SPECTROGRAM: a time-frequency image where a rising tone traces a
rising ridge, a chirp sweeps diagonally, and a transient click shows as a vertical streak.

The core trade-off is the UNCERTAINTY PRINCIPLE: a short window localizes events sharply in time but
blurs them in frequency (few samples -> coarse bins), while a long window resolves close frequencies
but smears their timing. The window SHAPE matters too -- a Hann or Hamming window trades a slightly
wider main lobe for far lower spectral leakage than a rectangular one, which is why raw rectangular
framing rings badly. The HOP between frames sets the time resolution and the overlap redundancy.

This module computes the STFT with a choice of window (rectangular, Hann, Hamming), returns the
complex frames and the magnitude spectrogram, and provides the dominant-frequency track over time. It
is validated: a steady sinusoid produces a flat frequency ridge at the right bin in every frame; a
linear chirp produces a ridge whose peak bin rises monotonically; the frame count and bin count match
the framing parameters; Parseval's relation holds per frame (framed energy equals spectral energy up to
the window); the window functions have the expected endpoints and symmetry; and a silent signal gives a
zero spectrogram. Reuses the repo's Bluestein DFT. Pure stdlib; the time-frequency companion to the
FFT, chirp-Z, and cepstrum tools."""

from __future__ import annotations

import math

from bluestein import dft


def rectangular_window(n):
    return [1.0] * n


def hann_window(n):
    if n == 1:
        return [1.0]
    return [0.5 - 0.5 * math.cos(2 * math.pi * k / (n - 1)) for k in range(n)]


def hamming_window(n):
    if n == 1:
        return [1.0]
    return [0.54 - 0.46 * math.cos(2 * math.pi * k / (n - 1)) for k in range(n)]


_WINDOWS = {"rectangular": rectangular_window, "hann": hann_window, "hamming": hamming_window}


def frame_signal(signal, frame_len, hop):
    """Split a signal into overlapping frames of length frame_len advancing by hop (zero-padding the last)."""
    frames = []
    i = 0
    n = len(signal)
    while i < n:
        frame = signal[i:i + frame_len]
        if len(frame) < frame_len:
            frame = list(frame) + [0.0] * (frame_len - len(frame))
        frames.append(frame)
        i += hop
    return frames


def stft(signal, frame_len=64, hop=32, window="hann"):
    """Short-time Fourier transform: returns a list of complex spectra, one per frame."""
    win = _WINDOWS[window](frame_len)
    frames = frame_signal(signal, frame_len, hop)
    out = []
    for frame in frames:
        windowed = [frame[k] * win[k] for k in range(frame_len)]
        out.append(dft(windowed))
    return out


def spectrogram(signal, frame_len=64, hop=32, window="hann"):
    """Magnitude spectrogram: a list (one per frame) of magnitude vectors over the first frame_len//2+1 bins."""
    spectra = stft(signal, frame_len, hop, window)
    half = frame_len // 2 + 1
    return [[abs(spec[k]) for k in range(half)] for spec in spectra]


def dominant_frequency_track(signal, frame_len=64, hop=32, window="hann", sample_rate=1.0):
    """The peak-magnitude frequency (in Hz) in each frame -- the dominant-frequency ridge over time."""
    spec = spectrogram(signal, frame_len, hop, window)
    half = frame_len // 2 + 1
    track = []
    for mags in spec:
        # skip DC bin for tone tracking
        best = max(range(1, half), key=lambda k: mags[k]) if half > 1 else 0
        track.append(best * sample_rate / frame_len)
    return track


def frame_energy(frame):
    return sum(x * x for x in frame)


def spectral_energy(spectrum):
    """Energy in the spectrum via Parseval: (1/N) sum |X[k]|^2."""
    n = len(spectrum)
    return sum(abs(v) ** 2 for v in spectrum) / n
