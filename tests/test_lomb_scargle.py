"""Tests for Lomb-Scargle: peaks at injected frequency (even + uneven), matches FFT, FAP behaviour."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lomb_scargle import (  # noqa: E402
    periodogram,
    best_frequency,
    best_period,
    frequency_grid,
    false_alarm_probability,
    fit_sinusoid,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fft import power_spectrum, frequencies as fft_freqs  # noqa: E402


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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def main():
    # ---- 1. even sampling: peak at injected frequency -----------------------------------
    f0 = 0.13
    times = [i * 0.5 for i in range(200)]  # dt = 0.5, so Nyquist = 1.0
    values = [math.sin(2 * math.pi * f0 * t) for t in times]
    freqs = frequency_grid(times, samples_per_peak=10)
    fbest, _ = best_frequency(times, values, freqs)
    check("even sampling: peak at injected f0", abs(fbest - f0) < 0.01, f"{fbest} vs {f0}")

    # ---- 2. uneven sampling: still peaks at f0 ------------------------------------------
    rng = _lcg(2024)
    t = 0.0
    utimes = []
    for _ in range(200):
        t += 0.1 + (rng() / (1 << 24)) * 0.9  # irregular gaps in [0.1, 1.0)
        utimes.append(t)
    uvalues = [math.sin(2 * math.pi * f0 * tt + 0.7) for tt in utimes]
    ufreqs = frequency_grid(utimes, samples_per_peak=10)
    ufbest, _ = best_frequency(utimes, uvalues, ufreqs)
    check("uneven sampling: peak at injected f0", abs(ufbest - f0) < 0.01, f"{ufbest} vs {f0}")

    # ---- 3. best_period is 1/f0 ---------------------------------------------------------
    check("best period ~ 1/f0", abs(best_period(times, values, freqs) - 1 / f0) < 0.5,
          f"{best_period(times, values, freqs)} vs {1/f0}")

    # ---- 4. two tones -> two peaks ------------------------------------------------------
    fa, fb = 0.08, 0.27
    vals2 = [math.sin(2 * math.pi * fa * tt) + 0.7 * math.sin(2 * math.pi * fb * tt) for tt in times]
    p = periodogram(times, vals2, freqs)
    # find the two largest local maxima
    peaks = sorted(range(1, len(freqs) - 1),
                   key=lambda i: p[i], reverse=True)
    top_freqs = []
    for i in peaks:
        if p[i] > p[i - 1] and p[i] > p[i + 1]:
            top_freqs.append(freqs[i])
        if len(top_freqs) == 2:
            break
    near_a = any(abs(f - fa) < 0.015 for f in top_freqs)
    near_b = any(abs(f - fb) < 0.015 for f in top_freqs)
    check("two tones produce peaks at both frequencies", near_a and near_b,
          f"peaks {[round(f,3) for f in top_freqs]}")

    # ---- 5. matches FFT on evenly-sampled data ------------------------------------------
    N = 128
    dt = 1.0
    ftest = 0.1  # cycles/sample, well inside Nyquist 0.5
    ev_times = [i * dt for i in range(N)]
    ev_vals = [math.sin(2 * math.pi * ftest * i) for i in range(N)]
    fft_f = fft_freqs(N, sample_rate=1.0 / dt)
    fft_p = power_spectrum(ev_vals)
    # FFT peak (positive freqs only)
    half = N // 2
    fft_peak_idx = max(range(1, half), key=lambda i: fft_p[i])
    fft_peak_f = fft_f[fft_peak_idx]
    ls_freqs = [i / (N * dt) for i in range(1, half)]
    ls_fbest, _ = best_frequency(ev_times, ev_vals, ls_freqs)
    check("LS peak freq matches FFT peak freq", abs(ls_fbest - fft_peak_f) < 1.5 / (N * dt),
          f"LS {ls_fbest:.4f} vs FFT {fft_peak_f:.4f}")

    # ---- 6. false alarm probability ------------------------------------------------------
    # pure noise: peak power modest, FAP near 1
    rng = _lcg(77)
    noise = [(rng() / (1 << 24)) - 0.5 for _ in range(200)]
    pn = periodogram(times, noise, freqs)
    peak_noise = max(pn)
    fap_noise = false_alarm_probability(peak_noise, len(freqs))
    # strong signal: high peak power, FAP near 0
    ps = periodogram(times, values, freqs)
    fap_sig = false_alarm_probability(max(ps), len(freqs))
    check("FAP high for pure noise", fap_noise > 0.01, f"{fap_noise:.3f}")
    check("FAP near zero for strong signal", fap_sig < 1e-6, f"{fap_sig:.2e}")
    check("signal peak power >> noise peak power", max(ps) > 3 * peak_noise,
          f"{max(ps):.1f} vs {peak_noise:.1f}")

    # ---- 7. amplitude/phase recovery ----------------------------------------------------
    amp0, phase0 = 2.5, 0.9
    vals3 = [amp0 * math.cos(2 * math.pi * f0 * tt + phase0) + 1.3 for tt in times]
    amp, phase, offset = fit_sinusoid(times, vals3, f0)
    check("recovered amplitude ~ injected", abs(amp - amp0) < 0.05, f"{amp:.3f} vs {amp0}")
    check("recovered offset ~ injected", abs(offset - 1.3) < 0.05, f"{offset:.3f}")
    # phase modulo 2pi
    dphase = (phase - phase0) % (2 * math.pi)
    check("recovered phase ~ injected", dphase < 0.05 or dphase > 2 * math.pi - 0.05,
          f"{phase:.3f} vs {phase0}")

    # ---- 8. edge cases ------------------------------------------------------------------
    check("FAP monotone in power", false_alarm_probability(10, 100) < false_alarm_probability(1, 100))
    try:
        frequency_grid([1.0])
        check("single time raises", False)
    except ValueError:
        check("single time raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
