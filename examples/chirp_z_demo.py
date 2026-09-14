"""Chirp-Z demo: zoom-FFT resolving two close tones a coarse FFT smears into one lump (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chirp_z as CZ
from bluestein import dft


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def main(outdir=None):
    fs = 32.0
    N = 256
    f1, f2 = 5.1, 5.4
    sig = [math.cos(2 * math.pi * f1 * n / fs) + math.cos(2 * math.pi * f2 * n / fs)
           for n in range(N)]

    # coarse FFT
    d = [abs(v) for v in dft(sig)]
    binw = fs / N
    # zoom into 4.8-5.8 Hz
    freqs, spec = CZ.zoom_fft(sig, 4.8, 5.8, 400, sample_rate=fs)
    mags = CZ.magnitude(spec)

    lines = []
    lines.append("Chirp Z-transform: zoom-FFT spectral analysis")
    lines.append("=" * 50)
    lines.append(f"signal: two tones at {f1} and {f2} Hz, fs={fs}, N={N}")
    lines.append(f"FFT bin width fs/N = {binw:.4f} Hz")
    lines.append("")
    lines.append("CZT samples the z-transform along any spiral z_k = A * W^-k, so choosing A, W")
    lines.append("on a narrow arc gives a fine-resolution DFT of just that band -- zoom-FFT.")
    lines.append("")
    mx = max(mags)
    peaks = [round(freqs[i], 3) for i in range(1, len(mags) - 1)
             if mags[i] > mags[i - 1] and mags[i] > mags[i + 1] and mags[i] > mx * 0.4]
    lines.append(f"zoom over 4.8-5.8 Hz with 400 points finds peaks at: {peaks}")
    lines.append(f"  (true tones {f1}, {f2} -- recovered to sub-bin accuracy)")
    lines.append("")
    lines.append("with A=1, W=exp(-2i*pi/N), M=N the CZT is exactly the ordinary DFT:")
    from bluestein import dft as _dft
    x = [1.0, 2.0, 3.0, 4.0]
    c = CZ.czt_as_dft(x)
    dd = _dft(x)
    lines.append(f"  max|CZT - DFT| on [1,2,3,4] = {max(abs(c[k]-dd[k]) for k in range(4)):.2e}")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 380
        ml, mt, w, h = 55, 60, 620, 130
        # top panel: coarse FFT bins over 4-7 Hz; bottom: zoom spectrum
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="26" fill="{TEXT}" font-size="15">'
                 f'Coarse FFT bins vs zoom-FFT over 4.8-5.8 Hz</text>')
        # coarse FFT: bins whose freq in [4,7]
        lo_f, hi_f = 4.0, 7.0
        binf = [(k * binw, d[k]) for k in range(N // 2) if lo_f <= k * binw <= hi_f]
        dmax = max(v for _, v in binf) or 1

        def sx(f, oy):
            return ml + (f - lo_f) / (hi_f - lo_f) * w

        # top: coarse FFT as stems
        oy1 = mt
        s.append(f'<text x="{ml}" y="{oy1-6}" fill="{GRAY}" font-size="11">'
                 f'coarse FFT (bin width {binw:.3f} Hz) -- peaks land between bins</text>')
        s.append(f'<line x1="{ml}" y1="{oy1+h}" x2="{ml+w}" y2="{oy1+h}" stroke="{GRAY}"/>')
        for f, v in binf:
            x = sx(f, oy1)
            s.append(f'<line x1="{x:.1f}" y1="{oy1+h:.1f}" x2="{x:.1f}" '
                     f'y2="{oy1+h-v/dmax*h:.1f}" stroke="{BLUE}" stroke-width="3"/>')
        # bottom: zoom spectrum as a curve
        oy2 = mt + h + 50
        zmax = max(mags) or 1
        s.append(f'<text x="{ml}" y="{oy2-6}" fill="{GRAY}" font-size="11">'
                 f'zoom-FFT (400 pts over 1 Hz) -- two peaks resolved</text>')
        s.append(f'<line x1="{ml}" y1="{oy2+h}" x2="{ml+w}" y2="{oy2+h}" stroke="{GRAY}"/>')

        def zx(f):
            return ml + (f - lo_f) / (hi_f - lo_f) * w
        pts = " ".join(f"{zx(freqs[i]):.1f},{oy2+h-mags[i]/zmax*h:.1f}" for i in range(len(freqs)))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{GREEN}" stroke-width="2"/>')
        for tf in (f1, f2):
            s.append(f'<line x1="{zx(tf):.1f}" y1="{oy2}" x2="{zx(tf):.1f}" y2="{oy2+h}" '
                     f'stroke="{RED}" stroke-width="1" stroke-dasharray="3,3"/>')
            s.append(f'<text x="{zx(tf):.1f}" y="{oy2-6}" fill="{RED}" font-size="9" '
                     f'text-anchor="middle">{tf}</text>')
        for f in (4, 5, 6, 7):
            s.append(f'<text x="{zx(f):.1f}" y="{oy2+h+16}" fill="{GRAY}" font-size="9" '
                     f'text-anchor="middle">{f} Hz</text>')
        s.append(f'<text x="{ml}" y="{H-8}" fill="{GRAY}" font-size="10">'
                 f'The fine CZT grid separates the two tones the coarse FFT lumps together.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "chirp_z.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
