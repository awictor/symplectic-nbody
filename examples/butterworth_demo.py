"""Demo: Butterworth IIR filter design -- maximally-flat low-pass, denoising a signal.

Designs low-pass filters of several orders, prints the frequency response, cleans a noisy two-tone
signal, and draws the magnitude responses (showing the flat passband and steeper roll-off with order)
plus the noisy-vs-filtered waveform.

    python examples/butterworth_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from butterworth import butter_lowpass, lfilter, filtfilt, magnitude, magnitude_db  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Butterworth: maximally-flat digital filters via the bilinear transform\n")

    fc = 0.1
    print(f"  low-pass magnitude at the cutoff fc={fc} (should be -3 dB = 0.7071):")
    for order in (2, 4, 8):
        b, a = butter_lowpass(order, fc)
        print(f"    order {order}: |H(fc)| = {magnitude(b, a, fc):.4f}  "
              f"({magnitude_db(b, a, fc):+.2f} dB);  "
              f"|H(2*fc)| = {magnitude_db(b, a, 2*fc):+.1f} dB,  "
              f"|H(4*fc)| = {magnitude_db(b, a, 4*fc):+.1f} dB")
    print()

    # denoise: a slow signal buried in high-frequency noise
    N = 600
    clean = [math.sin(2 * math.pi * 0.01 * i) for i in range(N)]
    rng = LCG(2024)
    noisy = [clean[i] + 0.4 * math.sin(2 * math.pi * 0.35 * i) + 0.15 * rng.normal()
             for i in range(N)]
    b, a = butter_lowpass(6, 0.05)
    filtered = filtfilt(b, a, noisy)

    # residual error vs the clean signal (middle section, avoiding edges)
    err_noisy = math.sqrt(sum((noisy[i] - clean[i]) ** 2 for i in range(100, 500)) / 400)
    err_filt = math.sqrt(sum((filtered[i] - clean[i]) ** 2 for i in range(100, 500)) / 400)
    print("  denoising a 0.01-cycle/sample sine buried in 0.35 tone + Gaussian noise:")
    print(f"    RMS error vs clean -- noisy: {err_noisy:.4f}   filtered: {err_filt:.4f}   "
          f"({err_noisy/err_filt:.1f}x cleaner)\n")

    print("  The passband is maximally flat (no ripple), the response is down exactly 3 dB at the")
    print("  cutoff, and each extra order adds 6 dB/octave of roll-off. filtfilt runs the filter")
    print("  forwards and backwards so the denoised signal has zero phase lag.")

    _svg(os.path.join(outdir, "butterworth.svg"), fc, noisy, filtered, clean)
    print(f"\n  wrote {os.path.join(outdir, 'butterworth.svg')}")


def _svg(path, fc, noisy, filtered, clean, width=760, height=520):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Butterworth low-pass: response (top) and denoising (bottom)</text>',
    ]

    # --- top panel: magnitude response of orders 2/4/8 ---
    ox, oy = 60, 250
    pw, ph = width - 100, 180
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<text x="{ox-8}" y="{oy-ph+4}" fill="#8b949e" font-size="10" text-anchor="end">1.0</text>')
    parts.append(f'<text x="{ox-8}" y="{oy+4}" fill="#8b949e" font-size="10" text-anchor="end">0</text>')

    freqs = [i / 300 * 0.5 for i in range(301)]

    def fx(f):
        return ox + f / 0.5 * pw

    def fy(m):
        return oy - m * ph

    colours = {2: "#4dabf7", 4: "#ffd43b", 8: "#06d6a0"}
    for order, col in colours.items():
        b, a = butter_lowpass(order, fc)
        pts = " ".join(f"{fx(f):.1f},{fy(magnitude(b,a,f)):.1f}" for f in freqs)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{ox+pw-90}" y="{oy-ph+18+18*(order//2-1)}" fill="{col}" '
                     f'font-size="11">order {order}</text>')
    # cutoff marker + -3dB line
    parts.append(f'<line x1="{fx(fc):.1f}" y1="{oy}" x2="{fx(fc):.1f}" y2="{oy-ph}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{fx(fc):.0f}" y="{oy+16}" fill="#ff6b6b" font-size="10" '
                 f'text-anchor="middle">fc</text>')
    parts.append(f'<line x1="{ox}" y1="{fy(0.7071):.1f}" x2="{ox+pw}" y2="{fy(0.7071):.1f}" '
                 f'stroke="#8b949e" stroke-width="0.7" stroke-dasharray="2 4"/>')
    parts.append(f'<text x="{ox+pw}" y="{fy(0.7071)-3:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">-3 dB</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+34}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">frequency (fraction of sample rate)</text>')

    # --- bottom panel: noisy vs filtered waveform ---
    bx, by = 60, 500
    bw, bh = width - 100, 190
    N = len(noisy)
    allv = noisy + filtered
    vmin, vmax = min(allv), max(allv)
    span = vmax - vmin or 1

    def wx(i):
        return bx + i / (N - 1) * bw

    def wy(v):
        return by - (v - vmin) / span * bh

    pnoisy = " ".join(f"{wx(i):.1f},{wy(noisy[i]):.1f}" for i in range(N))
    parts.append(f'<polyline points="{pnoisy}" fill="none" stroke="#8b949e" stroke-width="0.7" '
                 f'opacity="0.7"/>')
    pfilt = " ".join(f"{wx(i):.1f},{wy(filtered[i]):.1f}" for i in range(N))
    parts.append(f'<polyline points="{pfilt}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    parts.append(f'<text x="{bx}" y="{by-bh-6}" fill="#8b949e" font-size="11">'
                 f'grey: noisy input     green: filtered (zero-phase)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
