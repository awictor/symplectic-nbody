"""ESPRIT demo: super-resolve two close tones the FFT can't split, and beat Prony under noise."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import esprit_method as esprit
import prony


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


def make_signal(freqs, amps, damps, n, dt, phases=None):
    if phases is None:
        phases = [0.0] * len(freqs)
    x = []
    for m in range(n):
        v = 0.0
        t = m * dt
        for k in range(len(freqs)):
            v += amps[k] * math.exp(-damps[k] * t) * math.cos(2 * math.pi * freqs[k] * t + phases[k])
        x.append(v)
    return x


def main():
    lines = []
    lines.append("ESPRIT -- subspace frequency estimation beyond the FFT limit")
    lines.append("=" * 62)
    lines.append("")

    dt = 1.0 / 100.0     # 100 Hz sampling
    n = 100
    bin_hz = 1.0 / (n * dt)
    lines.append(f"Sampling 100 Hz, {n} samples -> FFT bin spacing = {bin_hz:.2f} Hz.")
    lines.append("")

    # super-resolution: two tones 0.3 bins apart
    f1, f2 = 20.0, 20.0 + 0.3 * bin_hz
    xs = make_signal([f1, f2], [1.0, 1.0], [0.0, 0.0], n, dt)
    npk, mags = esprit.periodogram_peak_count(xs, threshold=0.5)
    ms = esprit.esprit(xs, p=4, dt=dt)
    close = sorted(set(round(abs(f), 3) for f in ms["frequencies"] if 15 < abs(f) < 25))
    lines.append(f"Two tones at {f1:.2f} and {f2:.2f} Hz ({0.3:.1f} of a bin apart):")
    lines.append(f"  FFT periodogram: {npk} peak (they merge into one blob)")
    lines.append(f"  ESPRIT resolves: {close} Hz")
    lines.append("")

    # noise robustness vs Prony
    nN = 200
    fN = [8.0, 19.0]
    clean = make_signal(fN, [1.0, 0.8], [0.0, 0.0], nN, dt, phases=[0.3, 1.1])
    lines.append("Noise robustness (2 tones at 8 and 19 Hz, additive Gaussian noise):")
    lines.append("   SNR noise-amp    ESPRIT max-err   Prony max-err")
    lines.append("   " + "-" * 46)
    for amp in (0.0, 0.02, 0.05, 0.1, 0.2):
        rng = _R(2024)
        noisy = [clean[i] + amp * rng.normal() for i in range(nN)]
        me = esprit.esprit(noisy, p=4, dt=dt)
        fe = [abs(f) for f in me["frequencies"]]
        ee = max(abs(min(fe, key=lambda f: abs(f - t)) - t) for t in fN)
        try:
            mp = prony.prony(noisy, p=4, dt=dt)
            fp = [abs(f) for f in mp["frequencies"]]
            ep = max(abs(min(fp, key=lambda f: abs(f - t)) - t) for t in fN)
            eps = f"{ep:.4f}"
        except Exception:
            ep = float("inf")
            eps = "   fail"
        lines.append(f"   {amp:9.2f}      {ee:.4f}         {eps}")
    lines.append("")
    lines.append("  ESPRIT projects out the noise subspace before root-finding;")
    lines.append("  Prony's linear-prediction step has no such defense.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(xs, mags, ms, f1, f2, bin_hz, n, dt)
    return text, svg


def _svg(xs, mags, model, f1, f2, bin_hz, n, dt):
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'ESPRIT super-resolution: FFT blob vs two sharp lines</text>')

    # frequency axis in Hz over a zoom window around 20 Hz
    fmin, fmax = 15.0, 25.0
    x0, x1, y0, y1 = 55, 610, 70, 340

    def fx(f):
        return x0 + (f - fmin) / (fmax - fmin) * (x1 - x0)

    # periodogram magnitude curve (map bin index -> Hz)
    df = 1.0 / (n * dt)
    band = [(k * df, mags[k]) for k in range(len(mags)) if fmin <= k * df <= fmax]
    if band:
        mmax = max(v for _f, v in band) or 1.0

        def my(v):
            return y1 - (v / mmax) * (y1 - y0)

        pts = " ".join(f"{fx(f):.1f},{my(v):.1f}" for f, v in band)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["gray"]}" stroke-width="2"/>')
        parts.append(f'<text x="{x0}" y="{y0 - 8}" fill="{P["gray"]}" font-size="11">'
                     f'FFT periodogram (one smooth blob)</text>')

    # baseline
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')

    # true tone locations (green dashed verticals)
    for f, lab in ((f1, "f1"), (f2, "f2")):
        parts.append(f'<line x1="{fx(f):.1f}" y1="{y0}" x2="{fx(f):.1f}" y2="{y1}" '
                     f'stroke="{P["green"]}" stroke-width="1" stroke-dasharray="3,3"/>')

    # ESPRIT recovered frequencies (yellow sharp lines with markers)
    got = sorted(set(round(abs(f), 3) for f in model["frequencies"] if fmin < abs(f) < fmax))
    for f in got:
        parts.append(f'<line x1="{fx(f):.1f}" y1="{y0 - 5}" x2="{fx(f):.1f}" y2="{y1}" '
                     f'stroke="{P["yellow"]}" stroke-width="2"/>')
        parts.append(f'<circle cx="{fx(f):.1f}" cy="{y0 - 5:.1f}" r="4" fill="{P["yellow"]}"/>')
        parts.append(f'<text x="{fx(f):.1f}" y="{y0 - 12:.1f}" fill="{P["yellow"]}" '
                     f'font-size="10" text-anchor="middle">{f:.2f}</text>')

    # x ticks
    for f in (15, 17, 19, 20, 21, 23, 25):
        parts.append(f'<text x="{fx(f):.1f}" y="{y1 + 16:.1f}" fill="{P["gray"]}" '
                     f'font-size="10" text-anchor="middle">{f}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 + 34:.1f}" fill="{P["gray"]}" '
                 f'font-size="11" text-anchor="middle">frequency (Hz)</text>')

    parts.append(f'<text x="20" y="{H - 24}" fill="{P["green"]}" font-size="11">'
                 f'green dashed = true tones ({f1:.2f}, {f2:.2f} Hz, {0.3:.1f} bin apart)</text>')
    parts.append(f'<text x="20" y="{H - 8}" fill="{P["yellow"]}" font-size="11">'
                 f'yellow = ESPRIT lines -- two frequencies where the FFT sees one</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
