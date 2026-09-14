"""MUSIC demo: razor-sharp pseudospectrum peaks from noise-subspace orthogonality, and the eigenvalue split."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import music_spectrum as music


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


def tones(freqs, n, rng=None, noise=0.0):
    x = []
    for t in range(n):
        v = sum(math.cos(2 * math.pi * f * t) for f in freqs)
        if rng is not None and noise:
            v += noise * rng.normal()
        x.append(v)
    return x


def periodogram(x, n_freqs, fmax=0.5):
    n = len(x)
    freqs = [fmax * i / n_freqs for i in range(n_freqs)]
    out = []
    for fr in freqs:
        re = sum(x[t] * math.cos(-2 * math.pi * fr * t) for t in range(n))
        im = sum(x[t] * math.sin(-2 * math.pi * fr * t) for t in range(n))
        out.append((re * re + im * im) / n)
    return freqs, out


def main():
    lines = []
    lines.append("MUSIC -- MUltiple SIgnal Classification, subspace super-resolution")
    lines.append("=" * 66)
    lines.append("")

    n = 100
    bin_hz = 1.0 / n
    f1, f2 = 0.20, 0.20 + 0.5 * bin_hz
    rng = _R(5)
    x = tones([f1, f2], n, rng=rng, noise=0.05)
    lines.append(f"{n} samples, two tones at {f1:.4f} and {f2:.4f} cyc/sample")
    lines.append(f"({(f2 - f1)/bin_hz:.1f} of an FFT bin apart -- unresolvable by the FFT).")
    lines.append("")

    peaks = sorted(music.music_peaks(x, n_signal=4, m=40, n_peaks=2))
    lines.append(f"MUSIC peaks: {[round(p, 4) for p in peaks]}")
    lines.append(f"  true:      [{f1:.4f}, {f2:.4f}]")
    lines.append(f"  errors:    {[round(abs(peaks[i] - [f1, f2][i]), 5) for i in range(2)]}")
    lines.append("")

    # eigenvalue split
    ev = music.eigenvalue_spectrum(x, m=40)
    lines.append("Covariance eigenvalues (signal subspace = the few large ones):")
    lines.append("   " + "  ".join(f"{v:.3f}" for v in ev[:8]) + "  ...")
    lines.append(f"  4 signal eigenvalues then a flat noise floor at ~{ev[6]:.3f}")
    lines.append("")

    # peak-to-floor
    freqs_m, psd_m = music.music_spectrum(x, n_signal=4, m=40, n_freqs=2048)
    freqs_p, psd_p = periodogram(x, 2048)
    ratio_m = max(psd_m) / (sorted(psd_m)[len(psd_m) // 2])
    ratio_p = max(psd_p) / (sorted(psd_p)[len(psd_p) // 2])
    lines.append(f"Peak-to-median-floor ratio:  MUSIC {ratio_m:.0f}x   periodogram {ratio_p:.0f}x")
    lines.append("  -> MUSIC's peaks come from a projection going to zero, so they are razor-sharp.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(freqs_m, psd_m, freqs_p, psd_p, f1, f2, ev)
    return text, svg


def _svg(fm, psd_m, fp, psd_p, f1, f2, ev):
    W, H = 640, 460
    P = PALETTE
    fmin, fmax = 0.12, 0.30

    def band(freqs, psd):
        pts = [(freqs[i], psd[i]) for i in range(len(freqs)) if fmin <= freqs[i] <= fmax]
        mx = max(v for _f, v in pts) or 1.0
        return [(f, v / mx) for f, v in pts]

    bm = band(fm, psd_m)
    bp = band(fp, psd_p)

    x0, x1, y0, y1 = 55, 615, 55, 290

    def px(f):
        return x0 + (f - fmin) / (fmax - fmin) * (x1 - x0)

    def py(v):
        lo = 1e-3
        v = max(v, lo)
        return y1 - (math.log10(v) - math.log10(lo)) / (-math.log10(lo)) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'MUSIC pseudospectrum vs FFT: two tones half an FFT bin apart</text>')

    for f in (f1, f2):
        parts.append(f'<line x1="{px(f):.1f}" y1="{y0}" x2="{px(f):.1f}" y2="{y1}" '
                     f'stroke="{P["green"]}" stroke-width="1" stroke-dasharray="3,3"/>')

    def curve(pts, color, width):
        s = " ".join(f"{px(f):.1f},{py(v):.1f}" for f, v in pts)
        return f'<polyline points="{s}" fill="none" stroke="{color}" stroke-width="{width}"/>'

    parts.append(curve(bp, P["gray"], 1.8))
    parts.append(curve(bm, P["yellow"], 2.2))
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    for f in (0.12, 0.16, 0.20, 0.24, 0.28):
        parts.append(f'<text x="{px(f):.1f}" y="{y1 + 14:.1f}" fill="{P["gray"]}" '
                     f'font-size="10" text-anchor="middle">{f:.2f}</text>')
    parts.append(f'<text x="{x0}" y="{y0 - 6}" fill="{P["gray"]}" font-size="10">log-power</text>')
    parts.append(f'<text x="20" y="{y1 + 32}" fill="{P["yellow"]}" font-size="11">'
                 f'yellow = MUSIC (two razor peaks)</text>')
    parts.append(f'<text x="330" y="{y1 + 32}" fill="{P["gray"]}" font-size="11">'
                 f'gray = FFT periodogram (one blob)</text>')
    parts.append(f'<text x="20" y="{y1 + 48}" fill="{P["green"]}" font-size="11">'
                 f'green dashed = true tones</text>')

    # eigenvalue bar chart (signal vs noise split)
    bx0, bx1, by0, by1 = 55, 615, 360, 430
    top = ev[:16]
    mx = max(top) or 1.0
    bw = (bx1 - bx0) / len(top)
    parts.append(f'<text x="20" y="{by0 - 8}" fill="{P["text"]}" font-size="12">'
                 f'Covariance eigenvalues -- 4 signal (green) tower over the noise floor (gray)</text>')
    for i, v in enumerate(top):
        h = (v / mx) * (by1 - by0)
        col = P["green"] if i < 4 else P["gray"]
        parts.append(f'<rect x="{bx0 + i * bw + 1:.1f}" y="{by1 - h:.1f}" '
                     f'width="{bw - 2:.1f}" height="{h:.1f}" fill="{col}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
