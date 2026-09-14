"""Welch PSD demo: the jagged periodogram vs the smooth Welch estimate on the same noisy signal (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import welch_psd as WP


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
RED = "#ff6b6b"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main(outdir=None):
    rnd = _lcg(20260913)

    def gauss():
        return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())

    fs = 200
    N = 4096
    # two tones (25, 60 Hz) buried in white noise
    sig = [0.8 * math.cos(2 * math.pi * 25 * n / fs) + 0.6 * math.cos(2 * math.pi * 60 * n / fs)
           + 1.2 * gauss() for n in range(N)]

    fp, pp = WP.periodogram(sig, fs, window="hann")
    fw, pw = WP.welch(sig, segment_len=256, overlap=0.5, sample_rate=fs, window="hann")

    def cv(v):
        m = sum(v) / len(v)
        var = sum((x - m) ** 2 for x in v) / len(v)
        return math.sqrt(var) / m if m > 0 else 0.0

    lines = []
    lines.append("Welch's method: low-variance power spectral density")
    lines.append("=" * 54)
    lines.append(f"signal: 25 Hz + 60 Hz tones in white noise, fs={fs}, N={N}")
    lines.append("")
    lines.append(f"raw periodogram: 1 segment, {len(pp)} bins, coefficient of variation {cv(pp):.3f}")
    nseg = WP.num_segments(N, 256, 0.5)
    lines.append(f"Welch: {nseg} overlapping 256-sample Hann segments averaged, CV {cv(pw):.3f}")
    lines.append(f"  variance reduced ~{(cv(pp)/cv(pw))**2:.0f}x by averaging {nseg} segments")
    lines.append("")
    # peaks in the Welch PSD
    order = sorted(range(len(pw)), key=lambda k: -pw[k])[:2]
    peaks = sorted(round(fw[k], 1) for k in order)
    lines.append(f"Welch PSD peaks at {peaks} Hz (the two buried tones, cleanly resolved)")
    lines.append("")
    lines.append("The raw periodogram's variance never shrinks with more data -- it stays hairy.")
    lines.append("Welch trades a little frequency resolution for a dramatically smoother estimate.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 400
        ml, mt, w, h = 55, 55, 620, 140

        def logscale(psd):
            return [math.log10(p + 1e-9) for p in psd]

        def draw(freqs, psd, oy, col, title):
            lv = logscale(psd)
            lo, hi = min(lv), max(lv)
            rng = hi - lo or 1

            def sx(f):
                return ml + f / (fs / 2) * w

            def sy(v):
                return oy + h - (v - lo) / rng * h
            s.append(f'<text x="{ml}" y="{oy-6}" fill="{GRAY}" font-size="11">{title}</text>')
            s.append(f'<line x1="{ml}" y1="{oy+h}" x2="{ml+w}" y2="{oy+h}" stroke="{GRAY}"/>')
            pts = " ".join(f"{sx(freqs[k]):.1f},{sy(lv[k]):.1f}" for k in range(len(freqs)))
            s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.3"/>')
            # mark the true tone frequencies
            for tf in (25, 60):
                s.append(f'<line x1="{sx(tf):.1f}" y1="{oy}" x2="{sx(tf):.1f}" y2="{oy+h}" '
                         f'stroke="{YELLOW}" stroke-width="0.8" stroke-dasharray="3,3"/>')

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="26" fill="{TEXT}" font-size="15">'
                 f'PSD (log scale): jagged periodogram vs smooth Welch estimate</text>')
        draw(fp, pp, 55, RED, "raw periodogram (1 segment) -- hairy, high variance")
        draw(fw, pw, 235, GREEN, f"Welch ({nseg} averaged segments) -- smooth, tones stand out")
        for hz in (0, 25, 50, 60, 75, 100):
            x = ml + hz / (fs / 2) * w
            s.append(f'<text x="{x:.1f}" y="{H-14}" fill="{GRAY}" font-size="9" '
                     f'text-anchor="middle">{hz}</text>')
        s.append(f'<text x="{ml+w/2:.0f}" y="{H-2}" fill="{GRAY}" font-size="10" '
                 f'text-anchor="middle">frequency (Hz) -- yellow dashed = true tones 25 & 60</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "welch_psd.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
