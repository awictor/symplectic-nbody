"""Demo: the discrete wavelet transform -- multiresolution analysis and denoising.

Decomposes a signal with a transient spike into its multi-scale detail coefficients (showing where the
event is localised), denoises a noisy signal by thresholding details, and draws the coefficient pyramid
and the clean/noisy/denoised waveforms.

    python examples/wavelet_transform_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wavelet_transform import (dwt, idwt, denoise, coeffs_to_flat, energy)  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Discrete wavelet transform: seeing a signal at every scale at once\n")

    n = 128
    # a smooth wave with a localized transient burst around index 80
    signal = [math.sin(2 * math.pi * i / n) for i in range(n)]
    for i in range(78, 84):
        signal[i] += 2.0

    approx, details = dwt(signal, "db4")
    print(f"  {n}-point signal (smooth wave + transient burst near index 80), db4 wavelet:")
    print(f"    {len(details)} detail levels, sizes {[len(d) for d in details]}, "
          f"approx size {len(approx)}")
    # the finest level localizes the burst
    finest = details[0]
    peak = max(range(len(finest)), key=lambda i: abs(finest[i]))
    print(f"    finest-detail peak at coefficient {peak} -> signal index ~{peak*2} "
          f"(the burst is localised in time, unlike a Fourier spectrum)\n")

    # compaction
    flat = coeffs_to_flat(approx, details)
    mags = sorted((abs(c) for c in flat), reverse=True)
    for keep in (8, 16, 32):
        e = sum(m * m for m in mags[:keep]) / energy(flat)
        print(f"    top {keep:2d} of {n} coefficients hold {100*e:.1f}% of the energy")
    print()

    # denoising
    clean = [math.sin(2 * math.pi * i / n) + 0.5 * math.sin(8 * math.pi * i / n) for i in range(n)]
    rng = LCG(7)
    noisy = [clean[i] + 0.35 * (rng.u() - 0.5) for i in range(n)]
    den = denoise(noisy, "db4", threshold=0.12)
    err_noisy = math.sqrt(sum((noisy[i] - clean[i]) ** 2 for i in range(n)) / n)
    err_den = math.sqrt(sum((den[i] - clean[i]) ** 2 for i in range(n)) / n)
    print(f"  denoising by soft-thresholding detail coefficients:")
    print(f"    RMS error vs clean -- noisy {err_noisy:.4f}, denoised {err_den:.4f} "
          f"({err_noisy/err_den:.1f}x cleaner)\n")

    print("  A filter bank splits the signal into a coarse approximation and fine detail, then recurses")
    print("  on the approximation -- a pyramid that captures both what frequencies are present and WHERE.")
    print("  db4's two vanishing moments make smooth stretches nearly vanish, so energy piles into a few")
    print("  coefficients: the basis of wavelet compression and denoising.")

    _svg(os.path.join(outdir, "wavelet_transform.svg"), signal, details, clean, noisy, den)
    print(f"\n  wrote {os.path.join(outdir, 'wavelet_transform.svg')}")


def _svg(path, signal, details, clean, noisy, den, width=760, height=520):
    n = len(signal)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'Wavelet detail coefficients localise a transient (top); denoising (bottom)</text>',
    ]

    # top: signal + finest two detail levels stacked
    ox, top = 50, 55
    pw = width - 80
    rowh = 70

    def draw_series(series, y0, colour, label, scale_to=1.0):
        m = max(abs(v) for v in series) or 1.0
        h = rowh * 0.45
        mid = y0 + rowh / 2
        pts = " ".join(f"{ox + i/(len(series)-1)*pw:.1f},{mid - series[i]/m*h:.1f}"
                       for i in range(len(series)))
        return [f'<text x="{ox}" y="{y0+8:.0f}" fill="#8b949e" font-size="10">{label}</text>',
                f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.4"/>']

    parts += draw_series(signal, top, "#4dabf7", "signal (wave + burst)")
    parts += draw_series(details[0], top + rowh, "#ffd43b", "finest detail (locates the burst)")
    parts += draw_series(details[min(2, len(details)-1)], top + 2 * rowh, "#b197fc", "coarser detail")

    # bottom: clean / noisy / denoised overlaid
    by0 = top + 3 * rowh + 30
    bh = 150
    allv = clean + noisy + den
    vmin, vmax = min(allv), max(allv)
    span = vmax - vmin or 1

    def wy(v):
        return by0 + bh - (v - vmin) / span * bh

    def wx(i):
        return ox + i / (n - 1) * pw

    parts.append(f'<text x="{ox}" y="{by0-8:.0f}" fill="#8b949e" font-size="11">'
                 f'grey: noisy    orange: denoised (db4)    green: clean truth</text>')
    for series, col, w in ((noisy, "#8b949e", 0.7), (den, "#ff922b", 1.8), (clean, "#06d6a0", 1.2)):
        pts = " ".join(f"{wx(i):.1f},{wy(series[i]):.1f}" for i in range(n))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}" '
                     f'opacity="{0.7 if col=="#8b949e" else 1}"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
