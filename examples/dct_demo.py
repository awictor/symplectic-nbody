"""Demo: the discrete cosine transform -- energy compaction and JPEG-style lossy compression.

Transforms a smooth signal, shows how few coefficients hold nearly all its energy, reconstructs it from
a handful of coefficients (lossy compression), and draws the original vs the compressed reconstruction
plus the coefficient magnitudes.

    python examples/dct_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dct import dct, idct, dct2, idct2, energy, compaction_ratio  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Discrete cosine transform: energy compaction behind JPEG and MP3\n")

    n = 64
    signal = [3 * math.sin(2 * math.pi * i / n) + math.sin(6 * math.pi * i / n)
              + 0.5 * math.cos(2 * math.pi * i / n) for i in range(n)]

    X = dct(signal)
    print(f"  {n}-point smooth signal, energy compaction (fraction of energy in top-k coeffs):")
    for keep in (1, 2, 4, 8, 16):
        print(f"    top {keep:2d} of {n} coefficients: {100*compaction_ratio(signal, keep):.2f}%")
    print()

    # lossy compression: keep only the K largest coefficients
    print("  lossy compression by keeping the K largest coefficients:")
    for keep in (4, 8, 16):
        idx = set(sorted(range(n), key=lambda k: abs(X[k]), reverse=True)[:keep])
        Xc = [X[k] if k in idx else 0.0 for k in range(n)]
        recon = idct(Xc)
        rms = math.sqrt(sum((recon[i] - signal[i]) ** 2 for i in range(n)) / n)
        sig_rms = math.sqrt(energy(signal) / n)
        print(f"    keep {keep:2d}/{n} coeffs ({100*keep/n:.0f}% of data): RMS error {rms:.4f} "
              f"({100*rms/sig_rms:.2f}% of signal RMS)")
    print()

    # 2D: an 8x8 block, JPEG style
    block = [[128 + 60 * math.cos(math.pi * (i + j) / 8) for j in range(8)] for i in range(8)]
    D = dct2(block)
    # count coefficients above 1% of the DC magnitude
    dc = abs(D[0][0])
    signif = sum(1 for i in range(8) for j in range(8) if abs(D[i][j]) > 0.01 * dc)
    print(f"  8x8 image block: {signif} of 64 DCT coefficients exceed 1% of DC "
          f"-> most can be discarded\n")

    print("  Reflecting the signal evenly at its ends yields a real, boundary-smooth spectrum whose")
    print("  energy piles into the lowest frequencies. Keep the big low-frequency coefficients, drop")
    print("  the tiny high-frequency ones: lossy compression the eye and ear barely notice.")

    _svg(os.path.join(outdir, "dct.svg"), signal, X)
    print(f"\n  wrote {os.path.join(outdir, 'dct.svg')}")


def _svg(path, signal, X, width=760, height=460):
    n = len(signal)
    # reconstruction from 4 coefficients
    idx = set(sorted(range(n), key=lambda k: abs(X[k]), reverse=True)[:4])
    Xc = [X[k] if k in idx else 0.0 for k in range(n)]
    recon = idct(Xc)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'DCT energy compaction: signal (top), coefficients (bottom)</text>',
    ]

    # top panel: signal vs 4-coeff reconstruction
    ox, oy = 50, 230
    pw, ph = width - 90, 170
    allv = signal + recon
    vmin, vmax = min(allv), max(allv)
    span = vmax - vmin or 1

    def sx(i):
        return ox + i / (n - 1) * pw

    def sy(v):
        return oy - (v - vmin) / span * ph

    parts.append(f'<text x="{ox}" y="{oy-ph-6}" fill="#8b949e" font-size="12">'
                 f'blue: original signal    green: reconstructed from just 4 coefficients</text>')
    ps = " ".join(f"{sx(i):.1f},{sy(signal[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{ps}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    pr = " ".join(f"{sx(i):.1f},{sy(recon[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pr}" fill="none" stroke="#06d6a0" stroke-width="1.6" '
                 f'stroke-dasharray="5 3"/>')

    # bottom panel: coefficient magnitudes
    bx, by = 50, 440
    bw, bh = width - 90, 150
    mags = [abs(c) for c in X]
    mmax = max(mags) or 1
    barw = bw / n
    parts.append(f'<text x="{bx}" y="{by-bh-6}" fill="#8b949e" font-size="12">'
                 f'DCT coefficient magnitudes -- energy concentrated in the first few</text>')
    for k in range(n):
        h = bh * mags[k] / mmax
        col = "#ffd43b" if k in idx else "#8b949e"
        parts.append(f'<rect x="{bx + k*barw:.1f}" y="{by-h:.1f}" width="{barw-0.6:.1f}" '
                     f'height="{h:.1f}" fill="{col}"/>')
    parts.append(f'<text x="{bx+bw/2:.0f}" y="{by+18}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">coefficient index (low to high frequency); '
                 f'yellow = kept</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
