"""Tests for butterworth: -3dB at cutoff, monotone rolloff, tone separation, zero-phase filtfilt."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from butterworth import (butter_lowpass, butter_highpass, lfilter, filtfilt,  # noqa: E402
                         magnitude, magnitude_db, freq_response)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def tone_amplitude(signal, freq, n=None):
    """Estimate the amplitude of a tone at digital frequency ``freq`` by projecting onto sin/cos."""
    if n is None:
        n = len(signal)
    # use the tail to avoid the filter's startup transient
    s = signal[-n:]
    w = 2 * math.pi * freq
    c = sum(s[i] * math.cos(w * i) for i in range(len(s)))
    sn = sum(s[i] * math.sin(w * i) for i in range(len(s)))
    return 2 * math.sqrt(c * c + sn * sn) / len(s)


def main():
    # ---- 1. -3 dB at the cutoff, exactly, for many orders and cutoffs -----------------
    for order in (1, 2, 3, 4, 6, 8):
        for cutoff in (0.05, 0.1, 0.2, 0.35):
            b, a = butter_lowpass(order, cutoff)
            m = magnitude(b, a, cutoff)
            check(f"low-pass -3dB at cutoff (order={order}, fc={cutoff})",
                  abs(m - 1 / math.sqrt(2)) < 1e-6, f"|H|={m:.6f} (want {1/math.sqrt(2):.6f})")

    # ---- 2. DC gain ~ 1 for low-pass, Nyquist gain ~ 1 for high-pass ------------------
    b, a = butter_lowpass(4, 0.1)
    check("low-pass DC gain == 1", abs(magnitude(b, a, 0.0) - 1.0) < 1e-9,
          f"{magnitude(b, a, 0.0)}")
    check("low-pass Nyquist gain ~ 0", magnitude(b, a, 0.5) < 0.01, f"{magnitude(b, a, 0.5)}")

    bh, ah = butter_highpass(4, 0.1)
    check("high-pass -3dB at cutoff", abs(magnitude(bh, ah, 0.1) - 1 / math.sqrt(2)) < 1e-6)
    check("high-pass DC gain ~ 0", magnitude(bh, ah, 0.0) < 0.01, f"{magnitude(bh, ah, 0.0)}")
    check("high-pass Nyquist gain ~ 1", abs(magnitude(bh, ah, 0.5) - 1.0) < 1e-6)

    # ---- 3. monotone magnitude (no ripple -- the Butterworth signature) ---------------
    b, a = butter_lowpass(5, 0.15)
    freqs = [i / 400 * 0.5 for i in range(1, 401)]
    mags = [magnitude(b, a, f) for f in freqs]
    monotone = all(mags[i] >= mags[i + 1] - 1e-9 for i in range(len(mags) - 1))
    check("low-pass magnitude is monotone (no ripple)", monotone)

    # ---- 4. higher order -> steeper rolloff -------------------------------------------
    def rolloff_octave(order, fc=0.1):
        b, a = butter_lowpass(order, fc)
        # measure dB drop from one octave above cutoff to two octaves above
        f1, f2 = fc * 2, fc * 4
        return magnitude_db(b, a, f1) - magnitude_db(b, a, f2)
    r2 = rolloff_octave(2)
    r4 = rolloff_octave(4)
    check("higher order rolls off faster", r4 > r2, f"order2={r2:.1f}dB order4={r4:.1f}dB/oct")
    # theoretical asymptotic slope is 6*order dB/octave; deep in the stopband it approaches that
    b, a = butter_lowpass(4, 0.02)
    slope = magnitude_db(b, a, 0.08) - magnitude_db(b, a, 0.16)   # one octave in deep stopband
    check("order-4 asymptotic slope near 24 dB/octave", 20 < slope < 28, f"{slope:.1f} dB/oct")

    # ---- 5. tone separation: low-pass keeps the low tone, kills the high one ----------
    N = 2000
    f_low, f_high = 0.02, 0.30
    sig = [math.sin(2 * math.pi * f_low * i) + math.sin(2 * math.pi * f_high * i) for i in range(N)]
    b, a = butter_lowpass(6, 0.1)
    out = lfilter(b, a, sig)
    amp_low = tone_amplitude(out, f_low, 1000)
    amp_high = tone_amplitude(out, f_high, 1000)
    check("low-pass preserves low tone (~amplitude 1)", abs(amp_low - 1.0) < 0.1, f"{amp_low:.3f}")
    check("low-pass removes high tone", amp_high < 0.05, f"{amp_high:.3f}")

    # high-pass: opposite
    bh, ah = butter_highpass(6, 0.1)
    outh = lfilter(bh, ah, sig)
    check("high-pass removes low tone", tone_amplitude(outh, f_low, 1000) < 0.05)
    check("high-pass preserves high tone", abs(tone_amplitude(outh, f_high, 1000) - 1.0) < 0.1)

    # ---- 6. filtfilt zero phase: a pure passband tone comes back aligned --------------
    tone = [math.sin(2 * math.pi * 0.02 * i) for i in range(N)]
    b, a = butter_lowpass(4, 0.1)
    ff = filtfilt(b, a, tone)
    # cross-correlation peak at lag 0 (compare middle section to avoid edges)
    mid = slice(500, 1500)
    orig = tone[mid]
    filt = ff[mid]
    # phase lag: find lag maximizing correlation over small window
    best_lag, best_corr = 0, -1e18
    for lag in range(-5, 6):
        c = sum(orig[i] * filt[i + lag] for i in range(len(orig)) if 0 <= i + lag < len(filt))
        if c > best_corr:
            best_corr = c
            best_lag = lag
    check("filtfilt has zero phase lag", best_lag == 0, f"peak lag {best_lag}")

    # ---- 7. first-order low-pass coefficient sanity -----------------------------------
    # a first-order butterworth is a standard bilinear RC; b coeffs equal, a[0]=1
    b, a = butter_lowpass(1, 0.25)   # fc = fs/4 -> wc = tan(pi/4) = 1
    # with wc=1: pole at (1-1)/(1+1)=0, zero at -1; H(z) = k(1+z^-1)/(1+0) -> b=[0.5,0.5], a=[1,0]
    check("first-order fc=0.25 gives b=[0.5,0.5]", abs(b[0] - 0.5) < 1e-9 and abs(b[1] - 0.5) < 1e-9,
          f"b={b}")
    check("first-order fc=0.25 gives a=[1,0]", abs(a[0] - 1.0) < 1e-9 and abs(a[1]) < 1e-9, f"a={a}")

    # ---- 8. input validation ----------------------------------------------------------
    for bad in (0.0, 0.5, 0.7, -0.1):
        try:
            butter_lowpass(2, bad)
            check(f"cutoff {bad} rejected", False)
        except ValueError:
            check(f"cutoff {bad} rejected", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
