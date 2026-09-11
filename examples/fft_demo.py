"""Demo: the Fast Fourier Transform -- finding the frequencies in a signal.

Builds a two-tone signal, recovers its frequencies from the FFT magnitude spectrum (checked
against the naive DFT), and shows the O(n log n) vs O(n^2) operation-count gulf that makes the
FFT one of the most important algorithms ever written. Draws the signal and its spectrum, and
the cost comparison.

    python examples/fft_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fft import fft, dft, ifft, magnitude_spectrum, convolve, frequencies  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    n = 64
    # two tones at 4 Hz (amplitude 1) and 12 Hz (amplitude 0.5), sample rate n Hz
    sig = [math.cos(2 * math.pi * 4 * j / n) + 0.5 * math.cos(2 * math.pi * 12 * j / n)
           for j in range(n)]
    mag = magnitude_spectrum(sig)

    print("FFT: decompose a signal into its frequency components\n")
    print(f"  signal = cos(2pi*4t) + 0.5 cos(2pi*12t), {n} samples")
    print(f"  FFT matches the naive DFT: {all(abs(a - b) < 1e-9 for a, b in zip(fft(sig), dft(sig)))}")
    print(f"  round-trip ifft(fft(x)) == x: {all(abs(complex(a) - b) < 1e-9 for a, b in zip(sig, ifft(fft(sig))))}\n")
    print("  strongest frequency bins (0..n/2):")
    order = sorted(range(n // 2 + 1), key=lambda k: -mag[k])
    for k in order[:3]:
        print(f"    bin {k:>3} ({k} Hz): magnitude {mag[k]:.2f}")
    print("  -> peaks at 4 Hz and 12 Hz, amplitude ratio 2:1, exactly the input.\n")

    print("  Convolution via the FFT (multiply spectra): (1+2x)(3+4x+5x^2) =")
    print(f"    {[round(c, 4) for c in convolve([1, 2], [3, 4, 5])]}  (= 3 + 10x + 13x^2 + 10x^3)\n")

    print("  Operation count, FFT (n log2 n) vs naive DFT (n^2):")
    print(f"  {'n':>10}{'FFT':>14}{'DFT':>18}{'speedup':>10}")
    for e in (10, 15, 20):
        nn = 1 << e
        fft_ops = nn * e
        dft_ops = nn * nn
        print(f"  {nn:>10}{fft_ops:>14,}{dft_ops:>18,}{dft_ops // fft_ops:>9,}x")
    print("\n  For a million samples the FFT is ~50,000x faster -- the difference between")
    print("  instant and infeasible. It is why digital audio, JPEG, MP3, and MRI all exist.")

    _svg(os.path.join(outdir, "fft.svg"), sig, mag)
    print(f"\n  wrote {os.path.join(outdir, 'fft.svg')}")


def _svg(path, sig, mag, w=760, h=430):
    n = len(sig)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'FFT: a two-tone signal and its frequency spectrum</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the signal in time (top) decomposes into two sharp peaks in frequency (bottom)</text>',
    ]

    # top: the time-domain signal
    tx0, tx1 = 45, w - 30
    ty0, ty1 = 165, 65
    smax = max(abs(v) for v in sig) * 1.1

    def TX(i):
        return tx0 + i / (n - 1) * (tx1 - tx0)

    def TY(v):
        return (ty0 + ty1) / 2 - v / smax * ((ty0 - ty1) / 2)

    parts.append(f'<line x1="{tx0}" y1="{(ty0+ty1)/2:.1f}" x2="{tx1}" y2="{(ty0+ty1)/2:.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    pts = " ".join(f"{TX(i):.1f},{TY(v):.1f}" for i, v in enumerate(sig))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="1.8"/>')
    parts.append(f'<text x="{tx0:.1f}" y="{ty1-4:.1f}" fill="#8b949e" font-size="10">time domain</text>')

    # bottom: the magnitude spectrum (first half, real signal)
    bx0, bx1 = 45, w - 30
    by0, by1 = h - 55, 230
    half = n // 2
    mmax = max(mag[:half + 1]) * 1.1

    def BX(k):
        return bx0 + k / half * (bx1 - bx0)

    def BY(m):
        return by0 - m / mmax * (by0 - by1)

    parts.append(f'<line x1="{bx0}" y1="{by0}" x2="{bx1}" y2="{by0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{bx0}" y1="{by0}" x2="{bx0}" y2="{by1}" stroke="#8b949e" stroke-width="1.2"/>')
    bw = (bx1 - bx0) / half
    for k in range(half + 1):
        x = BX(k)
        col = "#06d6a0" if mag[k] > 0.1 * mmax else "#1f6f52"
        parts.append(f'<rect x="{x-bw/2+0.5:.1f}" y="{BY(mag[k]):.1f}" width="{max(bw-1,1):.1f}" '
                     f'height="{by0-BY(mag[k]):.1f}" fill="{col}"/>')
    # label the peaks
    for k in (4, 12):
        parts.append(f'<text x="{BX(k):.1f}" y="{BY(mag[k])-4:.1f}" fill="#06d6a0" font-size="9" '
                     f'text-anchor="middle">{k} Hz</text>')
    for k in (0, 16, 32):
        parts.append(f'<text x="{BX(k):.1f}" y="{by0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{(bx0+bx1)/2:.1f}" y="{by0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">frequency bin -> magnitude</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
