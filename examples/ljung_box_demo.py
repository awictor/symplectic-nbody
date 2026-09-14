"""Ljung-Box demo: tell white-noise residuals from leftover autocorrelation, with the ACF and confidence bands."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ljung_box as lb


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


def main():
    lines = []
    lines.append("Ljung-Box test -- is a series white noise, or is autocorrelation left over?")
    lines.append("=" * 74)
    lines.append("")

    n = 300
    rng = _R(2)
    wn = [rng.normal() for _ in range(n)]
    ar = [0.0]
    for _ in range(n):
        ar.append(0.7 * ar[-1] + rng.normal())
    ar = ar[1:]

    for name, data in (("white noise", wn), ("AR(1), phi=0.7", ar)):
        res = lb.ljung_box(data, lags=15)
        verdict = "WHITE NOISE (fail to reject)" if res["p_value"] > 0.05 else "AUTOCORRELATED (reject)"
        lines.append(f"{name}:")
        lines.append(f"  Q = {res['Q']:.2f} (df={res['df']}), p = {res['p_value']:.3e} -> {verdict}")
        lines.append(f"  first 5 autocorrelations: {[round(r,3) for r in res['autocorrelations'][:5]]}")
        lines.append("")

    # Q vs lags for both
    lines.append("Ljung-Box Q vs number of lags:")
    lines.append("   lags   white-noise Q   AR(1) Q")
    lines.append("   " + "-" * 34)
    for h in (5, 10, 15, 20, 30):
        qw = lb.ljung_box(wn, lags=h)["Q"]
        qa = lb.ljung_box(ar, lags=h)["Q"]
        lines.append(f"   {h:4d}   {qw:11.2f}   {qa:9.2f}")
    lines.append("")
    lines.append("White-noise Q tracks its df (~lags); AR(1) Q explodes -- the autocorrelation accumulates.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(wn, ar)
    return text, svg


def _svg(wn, ar):
    W, H = 640, 450
    P = PALETTE
    n = len(wn)
    band = 1.96 / math.sqrt(n)   # 95% white-noise confidence band for autocorrelations
    lags = 20

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Autocorrelation functions with 95% white-noise bands</text>')

    def acf_panel(data, y_top, label, color):
        rho = lb.autocorrelations(data, lags)[1:]
        x0, x1 = 55, 610
        h = 130
        y_zero = y_top + h / 2

        def px(k):
            return x0 + k / (lags - 1) * (x1 - x0)

        def py(r):
            return y_zero - r * (h / 2) / 1.0   # scale so |rho|=1 spans half-height

        # confidence band
        parts.append(f'<rect x="{x0}" y="{py(band):.1f}" width="{x1 - x0}" '
                     f'height="{py(-band) - py(band):.1f}" fill="{P["gray"]}" opacity="0.15"/>')
        parts.append(f'<line x1="{x0}" y1="{y_zero:.1f}" x2="{x1}" y2="{y_zero:.1f}" '
                     f'stroke="{P["gray"]}" stroke-width="1"/>')
        # stems
        for k in range(lags):
            r = rho[k]
            col = color if abs(r) > band else P["gray"]
            parts.append(f'<line x1="{px(k):.1f}" y1="{y_zero:.1f}" x2="{px(k):.1f}" '
                         f'y2="{py(r):.1f}" stroke="{col}" stroke-width="2.5"/>')
            parts.append(f'<circle cx="{px(k):.1f}" cy="{py(r):.1f}" r="2.5" fill="{col}"/>')
        parts.append(f'<text x="{x0}" y="{y_top - 4:.1f}" fill="{P["text"]}" font-size="12">{label}</text>')

    acf_panel(wn, 55, "white noise -- stems stay inside the band (no autocorrelation)", P["green"])
    acf_panel(ar, 250, "AR(1), phi=0.7 -- stems decay outside the band (autocorrelation)", P["red"])

    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'gray band = +/-1.96/sqrt(n); Ljung-Box pools these autocorrelations into one '
                 f'chi-squared test.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
