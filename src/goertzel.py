"""Goertzel algorithm: one DFT bin without the whole FFT, and how a phone decodes a keypad.

The FFT computes every frequency bin of a signal at once in O(n log n). But often you only care about
a FEW specific frequencies -- is there a 697 Hz tone in this phone audio? a 60 Hz hum? a pilot carrier?
Running a full FFT to read one bin is wasteful. The GOERTZEL algorithm (1958) computes a single DFT
bin in O(n) with a tiny second-order recurrence, using only real arithmetic until the final step, and
needing no buffer of the whole signal -- it can run sample-by-sample as data streams in. That makes it
the standard method for DTMF (touch-tone) decoding on cheap hardware, tone signalling, and
resonator-style filtering.

For a target bin k of an N-point transform, let omega = 2 pi k / N and coeff = 2 cos(omega). The
recurrence over the samples x[n] is

    s[n] = x[n] + coeff * s[n-1] - s[n-2],

and after the last sample the complex bin value is X_k = s[N-1] - exp(-i omega) s[N-2]. The
magnitude-squared, the only thing tone detection needs, is even cheaper:

    |X_k|^2 = s[N-1]^2 + s[N-2]^2 - coeff * s[N-1] s[N-2],

with no trig or complex numbers at all. This module computes the complex bin, the power at an
arbitrary (not necessarily integer) frequency, a bank of Goertzel detectors, and a full DTMF decoder
that maps the two strongest row/column tones to a keypad digit.

Validated against the DFT: the Goertzel complex bin matches a direct DFT bin to machine precision for
every integer k, the power matches |DFT|^2, a pure sinusoid peaks in exactly its own bin, off-target
frequencies read near zero, and the DTMF decoder recovers a dialled string from synthesized
dual-tone audio (including that a single tone or noise decodes to nothing). Pure stdlib; the
targeted-frequency companion to the FFT and the Levinson-Durbin / spectral tools."""

from __future__ import annotations

import cmath
import math


def goertzel_bin(x, k):
    """The complex DFT coefficient X_k of the N-point transform of x, by the Goertzel recurrence.
    k may be any real frequency index in [0, N)."""
    n = len(x)
    omega = 2 * math.pi * k / n
    coeff = 2 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in x:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    # X_k = s[N-1] - exp(-i omega) s[N-2], then a phase correction exp(i omega) to match the
    # standard DFT convention X_k = sum x[n] exp(-i 2 pi k n / N)
    y = complex(s_prev - s_prev2 * math.cos(omega), s_prev2 * math.sin(omega))
    return y * cmath.exp(1j * omega)


def goertzel_power(x, k):
    """|X_k|^2 by the cheap trig-free form of Goertzel."""
    n = len(x)
    omega = 2 * math.pi * k / n
    coeff = 2 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in x:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    return s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2


def goertzel_power_hz(x, freq, sample_rate):
    """Power at a physical frequency (Hz), converting to the fractional bin index k = f N / fs."""
    n = len(x)
    k = freq * n / sample_rate
    return goertzel_power(x, k)


def detector_bank(x, freqs, sample_rate):
    """Power at each frequency in `freqs` (Hz). Returns a dict {freq: power}."""
    return {f: goertzel_power_hz(x, f, sample_rate) for f in freqs}


# --- DTMF (touch-tone) ------------------------------------------------------
DTMF_ROWS = [697, 770, 852, 941]
DTMF_COLS = [1209, 1336, 1477, 1633]
DTMF_KEYS = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"],
]


def dtmf_tone(digit, duration, sample_rate):
    """Synthesize the dual-tone samples for a keypad digit."""
    for ri, row in enumerate(DTMF_KEYS):
        for ci, key in enumerate(row):
            if key == digit:
                fr = DTMF_ROWS[ri]
                fc = DTMF_COLS[ci]
                n = int(duration * sample_rate)
                return [math.sin(2 * math.pi * fr * i / sample_rate) +
                        math.sin(2 * math.pi * fc * i / sample_rate) for i in range(n)]
    raise ValueError(f"not a DTMF digit: {digit}")


def dtmf_decode(x, sample_rate, threshold_ratio=4.0):
    """Decode a single DTMF tone block to a keypad digit, or None if no clear dual tone is present.
    Picks the strongest row and column frequency; requires each to dominate the runner-up."""
    row_powers = detector_bank(x, DTMF_ROWS, sample_rate)
    col_powers = detector_bank(x, DTMF_COLS, sample_rate)

    def strongest(powers):
        ordered = sorted(powers.items(), key=lambda kv: -kv[1])
        top_f, top_p = ordered[0]
        second_p = ordered[1][1] if len(ordered) > 1 else 0.0
        return top_f, top_p, second_p

    rf, rp, rp2 = strongest(row_powers)
    cf, cp, cp2 = strongest(col_powers)
    # both a row and a column tone must clearly dominate
    if rp2 > 0 and rp / rp2 < threshold_ratio:
        return None
    if cp2 > 0 and cp / cp2 < threshold_ratio:
        return None
    # and there must be real energy (reject noise/silence relative to signal length)
    if rp < 1e-6 or cp < 1e-6:
        return None
    # a genuine dual tone has comparable row and column power; a single tone leaves the other
    # band near zero. Require the weaker of the two dominant tones to be within a factor.
    if max(rp, cp) / min(rp, cp) > 50:
        return None
    ri = DTMF_ROWS.index(rf)
    ci = DTMF_COLS.index(cf)
    return DTMF_KEYS[ri][ci]


# --- reference: direct DFT bin ----------------------------------------------
def dft_bin(x, k):
    """A single DFT coefficient computed directly (reference for Goertzel)."""
    n = len(x)
    return sum(x[m] * cmath.exp(-2j * math.pi * k * m / n) for m in range(n))
