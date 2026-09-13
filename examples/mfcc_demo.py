"""Demo: MFCC -- turning a rising chirp into the cepstral features a speech system would hear.

Synthesizes a frequency sweep (chirp), computes its MFCC feature matrix frame by frame, and draws two
panels: the triangular mel filterbank, and the MFCC coefficients over time as a heatmap that visibly
tracks the rising pitch.

    python examples/mfcc_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mfcc import mfcc, mel_filterbank, hz_to_mel, mel_to_hz  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("MFCC: mel-frequency cepstral coefficients, the front end of speech recognition\n")

    sr = 8000
    dur = 0.5
    n = int(sr * dur)
    f0, f1 = 300.0, 3000.0
    # linear chirp from f0 to f1
    signal = []
    for t in range(n):
        frac = t / n
        inst_f = f0 + (f1 - f0) * frac
        phase = 2 * math.pi * (f0 * (t / sr) + 0.5 * (f1 - f0) / dur * (t / sr) ** 2)
        signal.append(math.sin(phase))

    print(f"  input: {dur}s chirp sweeping {f0:.0f} Hz -> {f1:.0f} Hz at {sr} Hz sample rate")

    n_filters, n_coeffs = 26, 13
    feats = mfcc(signal, sr, frame_ms=25, hop_ms=10, n_filters=n_filters, n_coeffs=n_coeffs)
    print(f"  output: {len(feats)} frames x {n_coeffs} coefficients\n")

    # show mel scale spacing
    print(f"  mel scale warps frequency (fine low, coarse high):")
    print(f"    {'Hz':>8}{'mel':>10}")
    for f in [100, 500, 1000, 2000, 4000]:
        print(f"    {f:>8}{hz_to_mel(f):>10.1f}")

    # the first MFCC and a mid-order one over the sweep, to show the trend
    c0 = [row[0] for row in feats]
    c2 = [row[2] for row in feats]
    print(f"\n  MFCC[0] (log-energy) range over the sweep: {min(c0):.2f} .. {max(c0):.2f}")
    print(f"  MFCC[2] (spectral shape) range:            {min(c2):.2f} .. {max(c2):.2f}")
    print(f"\n  The heatmap's coefficient pattern shifts as the pitch rises -- exactly the")
    print(f"  compact, decorrelated representation a recognizer classifies on.")

    fb = mel_filterbank(n_filters, 256, sr)
    _svg(os.path.join(outdir, "mfcc.svg"), fb, feats, sr)
    print(f"\n  wrote {os.path.join(outdir, 'mfcc.svg')}")


def _svg(path, fb, feats, sr, width=760, height=460):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="24" fill="#e6edf3" font-size="14">'
        f'Mel filterbank (top) and MFCC coefficients over time (bottom, chirp 300-&gt;3000 Hz)</text>',
    ]

    # --- top panel: mel filterbank ---
    ox, oy, ow, oh = 55, 40, width - 90, 150
    n_bins = len(fb[0])
    colors = ["#4dabf7", "#ffd43b", "#06d6a0", "#ff922b", "#b197fc", "#ff6b6b"]

    def fx(k):
        return ox + ow * k / (n_bins - 1)

    def fy(v):
        return oy + oh * (1 - v)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for m, f in enumerate(fb):
        pts = " ".join(f"{fx(k):.1f},{fy(f[k]):.1f}" for k in range(n_bins))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[m % len(colors)]}" '
                     f'stroke-width="1"/>')
    nyq = sr / 2
    parts.append(f'<text x="{ox}" y="{oy+oh+13}" fill="#8b949e" font-size="9">0 Hz</text>')
    parts.append(f'<text x="{ox+ow:.0f}" y="{oy+oh+13}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">{nyq:.0f} Hz</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+13}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">FFT frequency bins -&gt; triangular mel filters</text>')

    # --- bottom panel: MFCC heatmap ---
    hx, hy, hw, hh = 55, 250, width - 90, 170
    n_frames = len(feats)
    n_coeffs = len(feats[0]) if feats else 0
    # normalize each coefficient across time for display
    vmin = min(v for row in feats for v in row)
    vmax = max(v for row in feats for v in row)
    rng = vmax - vmin + 1e-12
    cw = hw / n_frames
    ch = hh / n_coeffs
    for t, row in enumerate(feats):
        for c in range(n_coeffs):
            norm = (row[c] - vmin) / rng
            # viridis-ish: dark purple -> teal -> yellow
            r = int(40 + 200 * norm)
            g = int(30 + 200 * norm)
            b = int(120 + 80 * (1 - norm))
            x = hx + t * cw
            y = hy + c * ch
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw+0.6:.1f}" '
                         f'height="{ch+0.6:.1f}" fill="rgb({r},{g},{b})"/>')
    parts.append(f'<rect x="{hx}" y="{hy}" width="{hw}" height="{hh}" fill="none" stroke="#30363d"/>')
    parts.append(f'<text x="{hx-6}" y="{hy+8}" fill="#8b949e" font-size="9" text-anchor="end">c0</text>')
    parts.append(f'<text x="{hx-6}" y="{hy+hh:.0f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">c{n_coeffs-1}</text>')
    parts.append(f'<text x="{hx+hw/2:.0f}" y="{hy+hh+14:.0f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">time (frames) -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
