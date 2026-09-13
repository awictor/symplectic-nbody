"""Demo: Lomb-Scargle periodogram -- finding a period in unevenly-sampled data.

Simulates a variable star observed on irregular nights (a sinusoid at a known period plus noise and
big gaps), computes the Lomb-Scargle periodogram, recovers the period and its significance, and draws
the irregular light curve beside the power spectrum with the detected peak marked.

    python examples/lomb_scargle_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lomb_scargle import (  # noqa: E402
    periodogram,
    frequency_grid,
    best_frequency,
    false_alarm_probability,
    fit_sinusoid,
)


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Lomb-Scargle periodogram: periods from irregular samples, where the FFT can't go\n")

    rng = _lcg(2024)

    def rnd():
        return rng() / (1 << 24)

    # a variable star with true period P0 = 8.4 days, observed on scattered nights over ~200 days
    P0 = 8.4
    f0 = 1.0 / P0
    amp0 = 1.4
    times = []
    t = 0.0
    while t < 200:
        # observe most nights, skip some (weather/moon), with irregular gaps
        t += 0.4 + rnd() * 1.6
        if rnd() < 0.8:
            times.append(t)
    values = [amp0 * math.sin(2 * math.pi * f0 * tt + 1.1) + (rnd() - 0.5) * 0.5 for tt in times]

    print(f"  {len(times)} irregular observations over {times[-1]:.0f} days "
          f"(true period {P0} days).")

    freqs = frequency_grid(times, samples_per_peak=8)
    powers = periodogram(times, values, freqs)
    fbest, pbest = best_frequency(times, values, freqs)
    fap = false_alarm_probability(pbest, len(freqs))
    amp, phase, offset = fit_sinusoid(times, values, fbest)

    print(f"\n  Trial frequencies: {len(freqs)} from {freqs[0]:.4f} to {freqs[-1]:.4f} cycles/day")
    print(f"  Detected period:   {1/fbest:.4f} days  (peak power {pbest:.2f})")
    print(f"  False-alarm prob:  {fap:.2e}  (chance this peak is noise)")
    print(f"  Fitted amplitude:  {amp:.3f}  (injected {amp0})")

    print("\n  Top 3 periodogram peaks (period, power):")
    idx = sorted(range(1, len(freqs) - 1),
                 key=lambda i: powers[i], reverse=True)
    shown = 0
    seen = []
    for i in idx:
        if powers[i] > powers[i - 1] and powers[i] > powers[i + 1]:
            per = 1 / freqs[i]
            if all(abs(per - s) > 0.1 for s in seen):
                print(f"    P = {per:7.3f} days   power {powers[i]:.2f}")
                seen.append(per)
                shown += 1
        if shown == 3:
            break

    print("\n  Each trial frequency's power is how well a least-squares sinusoid at that frequency")
    print("  fits the irregular samples -- no grid, no interpolation. The tall peak is the period.")

    _svg(os.path.join(outdir, "lomb_scargle.svg"), times, values, freqs, powers, fbest)
    print(f"\n  wrote {os.path.join(outdir, 'lomb_scargle.svg')}")


def _svg(path, times, values, freqs, powers, fbest, width=760, height=440):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Top: irregular light curve. Bottom: Lomb-Scargle power (peak = period)</text>',
    ]

    # ---- top: light curve ---------------------------------------------------------------
    tx0, ty0, tw, th = 50, 50, width - 90, 150
    tmin, tmax = min(times), max(times)
    vmin, vmax = min(values), max(values)

    def tx(t):
        return tx0 + tw * (t - tmin) / (tmax - tmin or 1)

    def tvy(v):
        return ty0 + th * (1 - (v - vmin) / (vmax - vmin or 1))

    parts.append(f'<line x1="{tx0}" y1="{ty0+th}" x2="{tx0+tw}" y2="{ty0+th}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    for t, v in zip(times, values):
        parts.append(f'<circle cx="{tx(t):.1f}" cy="{tvy(v):.1f}" r="2.2" fill="#4dabf7" '
                     f'opacity="0.8"/>')
    parts.append(f'<text x="{tx0}" y="{ty0+th+18}" fill="#8b949e" font-size="9">time (days) -- '
                 f'note the gaps</text>')

    # ---- bottom: periodogram ------------------------------------------------------------
    px0, py0, pw, ph = 50, 250, width - 90, 150
    pmax = max(powers)

    def px(f):
        return px0 + pw * (f - freqs[0]) / (freqs[-1] - freqs[0] or 1)

    def pvy(p):
        return py0 + ph * (1 - p / (pmax or 1))

    parts.append(f'<line x1="{px0}" y1="{py0+ph}" x2="{px0+pw}" y2="{py0+ph}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    pts = " ".join(f"{px(freqs[i]):.1f},{pvy(powers[i]):.1f}" for i in range(len(freqs)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="1.5"/>')
    # mark the peak
    xb = px(fbest)
    parts.append(f'<line x1="{xb:.1f}" y1="{py0}" x2="{xb:.1f}" y2="{py0+ph}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{xb+4:.1f}" y="{py0+12}" fill="#ff6b6b" font-size="10">'
                 f'P = {1/fbest:.2f} d</text>')
    parts.append(f'<text x="{px0}" y="{py0+ph+18}" fill="#8b949e" font-size="9">'
                 f'frequency (cycles/day)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
