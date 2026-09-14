"""Wiener filtering: the minimum-mean-square-error way to pull a signal out of noise.

Given a signal corrupted by additive noise, y = x + n, what LINEAR filter recovers x with the smallest
possible mean-squared error? Norbert Wiener answered this in the 1940s, and the result is beautifully
simple in the frequency domain. If the signal has power spectrum S(f) and the noise has power spectrum
N(f), the optimal filter multiplies each frequency by

    H(f) = S(f) / (S(f) + N(f)),

a number between 0 and 1: where the signal dominates (S >> N) the filter passes the frequency almost
untouched (H -> 1); where noise dominates (N >> S) it attenuates hard (H -> 0). It is the
frequency-by-frequency answer to "how much do I trust this bin?" and is optimal among ALL linear
time-invariant filters for stationary signals -- the ancestor of modern spectral-subtraction denoisers,
deconvolution, and image restoration.

The same idea DECONVOLVES a blurred signal y = h * x + n: the regularized Wiener deconvolution divides
by the blur's transfer function but tempers the division where noise would blow up, X_hat = Y * conj(H)
/ (|H|^2 + N/S). This module implements Wiener denoising (given signal and noise power spectra, or a
noise variance) and Wiener deconvolution, in the frequency domain via the repo's Bluestein DFT. It is
validated: the filter gain H lies in [0, 1] and equals 1 when there is no noise and ~0 when the signal
vanishes; denoising a noisy sinusoid raises the signal-to-noise ratio; a clean signal is passed almost
unchanged; the recovered estimate has lower mean-squared error than the noisy input; deconvolution
recovers a signal blurred by a known kernel far better than naive inverse filtering in the presence of
noise; and the gain is monotone in the local SNR. Reuses the repo's Bluestein DFT. Pure stdlib; the
signal-restoration companion to the FFT, cepstrum, and Savitzky-Golay tools."""

from __future__ import annotations

from bluestein import dft, idft, convolve


def wiener_gain(signal_power, noise_power):
    """The Wiener gain H = S / (S + N) per frequency bin (each in [0, 1])."""
    return [s / (s + nz) if (s + nz) > 0 else 0.0 for s, nz in zip(signal_power, noise_power)]


def denoise(y, signal_power, noise_power):
    """Wiener-denoise y given per-bin signal and noise power spectra. Returns the real estimate."""
    Y = dft(y)
    H = wiener_gain(signal_power, noise_power)
    X = [Y[k] * H[k] for k in range(len(Y))]
    x = idft(X)
    return [v.real for v in x]


def denoise_stationary(y, noise_var):
    """Wiener-denoise y assuming white noise of the given variance; estimate the signal power from y.

    A practical estimator: |Y|^2 / n is the noisy periodogram; subtract the noise floor to estimate S."""
    n = len(y)
    Y = dft(y)
    noise_power_bin = noise_var * n               # white noise: flat power spectrum, total = var*n per bin scale
    # estimate signal power per bin: max(|Y|^2 - noise_floor, small)
    sig_power = [max(abs(Y[k]) ** 2 - noise_power_bin, 1e-12) for k in range(n)]
    H = [sig_power[k] / (sig_power[k] + noise_power_bin) for k in range(n)]
    X = [Y[k] * H[k] for k in range(n)]
    x = idft(X)
    return [v.real for v in x]


def deconvolve(y, kernel, noise_to_signal=0.01):
    """Regularized Wiener deconvolution: recover x from y = kernel * x + noise.

    noise_to_signal is the N/S ratio used to temper the inverse. Returns the real estimate (length of y)."""
    n = len(y)
    # zero-pad the kernel to length n
    h = list(kernel) + [0.0] * (n - len(kernel))
    Y = dft(y)
    H = dft(h)
    X = []
    for k in range(n):
        denom = abs(H[k]) ** 2 + noise_to_signal
        X.append(Y[k] * H[k].conjugate() / denom if denom > 0 else 0.0)
    x = idft(X)
    return [v.real for v in x]


def snr(clean, noisy):
    """Signal-to-noise ratio in dB of `noisy` relative to `clean`."""
    import math
    sig = sum(c * c for c in clean)
    err = sum((noisy[i] - clean[i]) ** 2 for i in range(len(clean)))
    if err <= 0:
        return float("inf")
    return 10 * math.log10(sig / err)


def mse(a, b):
    """Mean squared error between two equal-length sequences."""
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) / len(a)
