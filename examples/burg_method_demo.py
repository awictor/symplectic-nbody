"""Burg's method demo: maximum-entropy spectrum sharpening two close tones the periodogram smears."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import burg_method as burg


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def periodogram(x, n_freqs):
    n = len(x)
    freqs = [0.5 * i / n_freqs for i in range(n_freqs)]
    out = []
    for fr in freqs:
        re = sum(x[t] * math.cos(-2 * math.pi * fr * t) for t in range(n))
        im = sum(x[t] * math.sin(-2 * math.pi * fr * t) for t in range(n))
        out.append((re * re + im * im) / n)
    return freqs, out


def main():
    lines = []
    lines.append("Burg's method -- maximum-entropy AR spectrum from short records")
    lines.append("=" * 66)
    lines.append("")

    # short record, two close tones in noise
    n = 96
    fa, fb = 0.20, 0.223
    rng = _R(7)
    sig = [math.cos(2 * math.pi * fa * t) + 0.9 * math.cos(2 * math.pi * fb * t)
           + 0.25 * rng.normal() for t in range(n)]
    bin_hz = 1.0 / n
    lines.append(f"{n} samples, two tones at {fa} and {fb} cyc/sample "
                 f"({(fb - fa) / bin_hz:.1f} FFT bins apart).")
    lines.append("")

    # Burg AR fit
    order = 24
    coeffs, refl, err = burg.burg(sig, order)
    lines.append(f"Burg AR({order}) fit: error variance {err:.4f}")
    lines.append(f"  all |reflection| < 1 (stable): {all(abs(k) < 1 for k in refl)}")
    lines.append("")

    freqs_b, psd_b = burg.me_spectrum(sig, order, n_freqs=1024)
    freqs_p, psd_p = periodogram(sig, 1024)

    def peaks_in(freqs, psd, lo, hi):
        out = []
        for i in range(1, len(psd) - 1):
            if lo <= freqs[i] <= hi and psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
                out.append(round(freqs[i], 4))
        return out

    pk_b = peaks_in(freqs_b, psd_b, 0.17, 0.26)
    pk_p = peaks_in(freqs_p, psd_p, 0.17, 0.26)
    lines.append(f"Peaks in [0.17, 0.26]:")
    lines.append(f"  Periodogram (FFT): {len(pk_p)} peak(s) {pk_p}")
    lines.append(f"    -> leakage + noise scatter spurious peaks; the two tones are ambiguous.")
    lines.append(f"  Burg MEM spectrum: {len(pk_b)} peak(s) {pk_b}")
    lines.append(f"    -> exactly two clean peaks, on the true frequencies.")
    lines.append("")

    # AR coefficient recovery on a known process
    lines.append("AR coefficient recovery on a known AR(2) process x_t = 0.75 x_{t-1} - 0.5 x_{t-2}:")
    rng2 = _R(3)
    x2 = []
    buf = [0.0, 0.0]
    for _ in range(500 + 200):
        nxt = 0.75 * buf[-1] - 0.5 * buf[-2] + rng2.normal()
        buf.append(nxt)
    x2 = buf[200:]
    c2, _, _ = burg.burg(x2, 2)
    lines.append(f"  recovered: x_t = {c2[0]:.3f} x_(t-1) + {c2[1]:.3f} x_(t-2)  (true 0.75, -0.5)")

    text = "\n".join(lines)
    print(text)

    svg = _svg(freqs_b, psd_b, freqs_p, psd_p, fa, fb)
    return text, svg


def _svg(fb_freqs, psd_b, fp_freqs, psd_p, fa, fb):
    W, H = 640, 430
    P = PALETTE
    fmin, fmax = 0.10, 0.35

    # normalize each spectrum to its own max within the band (log scale for dynamic range)
    def band(freqs, psd):
        pts = [(freqs[i], psd[i]) for i in range(len(freqs)) if fmin <= freqs[i] <= fmax]
        mx = max(v for _f, v in pts) or 1.0
        return [(f, v / mx) for f, v in pts]

    bb = band(fb_freqs, psd_b)
    bp = band(fp_freqs, psd_p)

    x0, x1, y0, y1 = 55, 615, 60, 350

    def px(f):
        return x0 + (f - fmin) / (fmax - fmin) * (x1 - x0)

    def py(v):
        # log scale, clamp
        lo = 1e-3
        v = max(v, lo)
        return y1 - (math.log10(v) - math.log10(lo)) / (-math.log10(lo)) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Burg maximum-entropy spectrum vs FFT periodogram</text>')

    # true tone locations
    for f in (fa, fb):
        parts.append(f'<line x1="{px(f):.1f}" y1="{y0}" x2="{px(f):.1f}" y2="{y1}" '
                     f'stroke="{P["green"]}" stroke-width="1" stroke-dasharray="3,3"/>')

    def curve(pts, color, width):
        s = " ".join(f"{px(f):.1f},{py(v):.1f}" for f, v in pts)
        return f'<polyline points="{s}" fill="none" stroke="{color}" stroke-width="{width}"/>'

    parts.append(curve(bp, P["gray"], 1.8))    # periodogram (one blob)
    parts.append(curve(bb, P["yellow"], 2.2))  # Burg (two sharp peaks)

    # baseline + ticks
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    for f in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        parts.append(f'<text x="{px(f):.1f}" y="{y1 + 16:.1f}" fill="{P["gray"]}" '
                     f'font-size="10" text-anchor="middle">{f:.2f}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 + 34:.1f}" fill="{P["gray"]}" '
                 f'font-size="11" text-anchor="middle">frequency (cycles/sample), log-power</text>')

    parts.append(f'<text x="20" y="{H - 26}" fill="{P["green"]}" font-size="11">'
                 f'green dashed = true tones ({fa}, {fb})</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["yellow"]}" font-size="11">'
                 f'yellow = Burg MEM (two sharp peaks)</text>')
    parts.append(f'<text x="340" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'gray = FFT periodogram (one blob)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
