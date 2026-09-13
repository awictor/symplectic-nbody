"""MFCC: the mel-frequency cepstral coefficients that turn sound into the features speech systems hear.

Nearly every speech recognizer, speaker-ID system, and music classifier of the last few decades begins
by reducing a slice of audio to a short vector of MEL-FREQUENCY CEPSTRAL COEFFICIENTS. MFCCs are
engineered to mimic how human hearing works and to throw away exactly what does not matter for
recognition, and the pipeline is a lovely chain of signal-processing ideas:

  1. FRAME the signal into short overlapping windows (speech is quasi-stationary for ~25 ms) and taper
     each frame with a Hamming window to suppress spectral leakage.
  2. POWER SPECTRUM of each frame via the FFT -- how much energy sits at each frequency.
  3. MEL FILTERBANK. The ear resolves low frequencies finely and high frequencies coarsely, so we warp
     to the MEL SCALE (mel = 2595 log10(1 + f/700)) and sum the power under a bank of TRIANGULAR
     filters spaced evenly in mel. Each filter reports the energy in one perceptual band.
  4. LOG of each band energy -- loudness is perceived logarithmically (decibels), and the log turns the
     multiplicative source-filter model of speech into an additive one.
  5. DISCRETE COSINE TRANSFORM of the log-mel-energies. The DCT DECORRELATES the bands (adjacent mel
     bands are highly correlated) and compacts the information into the first few coefficients; those
     low-order coefficients are the MFCCs. Coefficient 0 is overall log-energy; the next dozen capture
     the spectral SHAPE (the vocal-tract filter) while discarding pitch and fine detail.

This module implements the mel/hz conversions, the triangular mel filterbank, framing with a Hamming
window, and the full MFCC computation, reusing the repo's FFT and DCT. It is validated by construction
and against references: mel<->hz round-trips and is monotonic; the filterbank is triangular,
non-negative, and its filters overlap to cover the band (a partition-like sum); a pure tone lights up
the mel band containing its frequency and (via a brute-force DFT power spectrum) matches the FFT power;
the log-mel/DCT stage matches a direct DCT of the log energies; MFCC vectors of two well-separated
tones differ while repeats of the same tone match; and coefficient 0 tracks overall energy. Pure
stdlib; the audio-feature companion to the FFT, the DCT, and the spectrogram tools."""

from __future__ import annotations

import math

from fft import fft
from dct import dct


def hz_to_mel(f):
    """Convert a frequency in Hz to the mel scale (O'Shaughnessy / HTK form)."""
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    """Convert a mel value back to Hz."""
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_filters, n_fft, sample_rate, fmin=0.0, fmax=None):
    """Triangular mel filterbank: a list of n_filters weight vectors over the n_fft/2+1 FFT bins.

    Each filter is a triangle in frequency, peaking at its centre mel-spaced frequency and falling to
    zero at its neighbours' centres.
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    n_bins = n_fft // 2 + 1

    # n_filters+2 mel points equally spaced between fmin and fmax
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = [mel_min + (mel_max - mel_min) * i / (n_filters + 1) for i in range(n_filters + 2)]
    hz_points = [mel_to_hz(m) for m in mel_points]
    # FFT bin index of each hz point
    bin_points = [int(math.floor((n_fft + 1) * h / sample_rate)) for h in hz_points]

    filters = []
    for m in range(1, n_filters + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        f = [0.0] * n_bins
        for k in range(left, center):
            if center > left and 0 <= k < n_bins:
                f[k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center and 0 <= k < n_bins:
                f[k] = (right - k) / (right - center)
        if 0 <= center < n_bins:
            f[center] = 1.0
        filters.append(f)
    return filters


def hamming_window(n):
    """Hamming window of length n."""
    if n == 1:
        return [1.0]
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def frame_signal(signal, frame_len, hop):
    """Split a signal into overlapping frames of length frame_len advanced by hop (zero-padded tail)."""
    frames = []
    i = 0
    n = len(signal)
    while i < n:
        frame = signal[i:i + frame_len]
        if len(frame) < frame_len:
            frame = list(frame) + [0.0] * (frame_len - len(frame))
        frames.append(list(frame))
        i += hop
        if i + frame_len > n and i < n:
            # include a final tail frame then stop
            last = signal[i:i + frame_len]
            if last:
                frames.append(list(last) + [0.0] * (frame_len - len(last)))
            break
    return frames


def _pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def power_spectrum(frame):
    """Single-sided power spectrum of a frame via the FFT (length n_fft/2+1)."""
    n_fft = _pow2(len(frame))
    padded = list(frame) + [0.0] * (n_fft - len(frame))
    spec = fft(padded)
    n_bins = n_fft // 2 + 1
    return [(spec[k].real ** 2 + spec[k].imag ** 2) / n_fft for k in range(n_bins)], n_fft


def mfcc_frame(frame, sample_rate, n_filters=26, n_coeffs=13, filterbank=None):
    """MFCC vector of a single frame: mel filterbank -> log -> DCT, keeping n_coeffs coefficients."""
    win = hamming_window(len(frame))
    windowed = [frame[i] * win[i] for i in range(len(frame))]
    ps, n_fft = power_spectrum(windowed)
    fb = filterbank if filterbank is not None else mel_filterbank(n_filters, n_fft, sample_rate)
    # energy in each mel band
    mel_energies = []
    for f in fb:
        e = sum(f[k] * ps[k] for k in range(len(ps)))
        mel_energies.append(math.log(e + 1e-12))
    coeffs = dct(mel_energies)
    return coeffs[:n_coeffs]


def mfcc(signal, sample_rate, frame_ms=25.0, hop_ms=10.0, n_filters=26, n_coeffs=13):
    """MFCC feature matrix: one row of n_coeffs coefficients per frame of the signal."""
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    frames = frame_signal(signal, frame_len, hop)
    if not frames:
        return []
    n_fft = _pow2(frame_len)
    fb = mel_filterbank(n_filters, n_fft, sample_rate)
    return [mfcc_frame(f, sample_rate, n_filters, n_coeffs, filterbank=fb) for f in frames]


def brute_power_spectrum(frame):
    """Reference single-sided power spectrum by a direct DFT (no FFT)."""
    n = len(frame)
    n_fft = _pow2(n)
    padded = list(frame) + [0.0] * (n_fft - n)
    n_bins = n_fft // 2 + 1
    ps = []
    for k in range(n_bins):
        re = im = 0.0
        for t in range(n_fft):
            ang = -2 * math.pi * k * t / n_fft
            re += padded[t] * math.cos(ang)
            im += padded[t] * math.sin(ang)
        ps.append((re * re + im * im) / n_fft)
    return ps
