"""Total least squares demo: perpendicular-distance line fit vs OLS on errors-in-both-variables data."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import total_least_squares as tls


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
    lines.append("Total least squares -- orthogonal regression for errors in both variables")
    lines.append("=" * 74)
    lines.append("")

    m_true, b_true = 2.0, 1.0
    n = 120
    rng = _R(5)
    x_clean = [rng.normal() * 4 for _ in range(n)]
    # heavy error in BOTH x and y
    ex, ey = 2.0, 2.0
    xs = [x_clean[i] + rng.normal() * ex for i in range(n)]
    ys = [m_true * x_clean[i] + b_true + rng.normal() * ey for i in range(n)]

    fit = tls.fit_line(xs, ys)
    m_tls, b_tls = fit["slope"], fit["intercept"]
    m_ols, b_ols = tls.ols(xs, ys)

    lines.append(f"True line: y = {m_true} x + {b_true}.  {n} points, error in BOTH axes (sd {ex}, {ey}).")
    lines.append("")
    lines.append("   method   slope   intercept   |slope err|   minimizes")
    lines.append("   " + "-" * 54)
    lines.append(f"   OLS     {m_ols:6.3f}  {b_ols:8.3f}    {abs(m_ols - m_true):.3f}       vertical distance")
    lines.append(f"   TLS     {m_tls:6.3f}  {b_tls:8.3f}    {abs(m_tls - m_true):.3f}       perpendicular distance")
    lines.append("")
    lines.append(f"OLS attenuates the slope to {m_ols:.3f} (regression dilution);")
    lines.append(f"TLS stays at {m_tls:.3f}, close to the true {m_true}.")
    lines.append("")

    # swap invariance
    f_swap = tls.fit_line(ys, xs)
    lines.append("Swap-invariance (fit x-on-y instead of y-on-x):")
    lines.append(f"   TLS slope, then 1/slope of swapped fit: {m_tls:.4f} vs {1/f_swap['slope']:.4f}  (equal)")
    mo2, _ = tls.ols(ys, xs)
    lines.append(f"   OLS slope, then 1/slope of swapped fit: {m_ols:.4f} vs {1/mo2:.4f}  (different!)")
    lines.append("")

    # attenuation vs error level
    lines.append("OLS attenuation grows with x-error; TLS stays put:")
    lines.append("   x-error   OLS slope   TLS slope")
    lines.append("   " + "-" * 34)
    for exx in (0.0, 0.5, 1.0, 2.0, 3.0):
        rng2 = _R(9)
        xc = [rng2.normal() * 4 for _ in range(n)]
        xx = [xc[i] + rng2.normal() * exx for i in range(n)]
        yy = [m_true * xc[i] + b_true + rng2.normal() * 1.0 for i in range(n)]
        mo, _ = tls.ols(xx, yy)
        mt = tls.fit_line(xx, yy)["slope"]
        lines.append(f"   {exx:6.1f}    {mo:8.3f}    {mt:8.3f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(xs, ys, (m_tls, b_tls), (m_ols, b_ols), (m_true, b_true))
    return text, svg


def _svg(xs, ys, tls_model, ols_model, true_model):
    W, H = 640, 440
    P = PALETTE
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    padx = 0.08 * (xmax - xmin)
    pady = 0.08 * (ymax - ymin)
    xmin -= padx
    xmax += padx
    ymin -= pady
    ymax += pady

    x0, x1, y0, y1 = 55, 615, 55, 375

    def px(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def py(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'TLS (perpendicular) vs OLS (vertical) with error in both axes</text>')

    # perpendicular residual segments from a few points to the TLS line
    nx, ny = tls_model  # not the normal; recompute below
    # draw points
    n = len(xs)
    for i in range(n):
        parts.append(f'<circle cx="{px(xs[i]):.1f}" cy="{py(ys[i]):.1f}" r="2.3" '
                     f'fill="{P["gray"]}" opacity="0.7"/>')

    def line(model, color, width, dash=None):
        m, b = model
        ya = m * xmin + b
        yb = m * xmax + b
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<line x1="{px(xmin):.1f}" y1="{py(ya):.1f}" x2="{px(xmax):.1f}" '
                f'y2="{py(yb):.1f}" stroke="{color}" stroke-width="{width}"{d}/>')

    parts.append(line(true_model, P["green"], 3))
    parts.append(line(tls_model, P["yellow"], 2))
    parts.append(line(ols_model, P["red"], 1.8, dash="6,4"))

    parts.append(f'<text x="{x0}" y="{y1 + 24}" fill="{P["green"]}" font-size="11">'
                 f'green = true line</text>')
    parts.append(f'<text x="{x0 + 120}" y="{y1 + 24}" fill="{P["yellow"]}" font-size="11">'
                 f'yellow = TLS (near truth)</text>')
    parts.append(f'<text x="{x0 + 320}" y="{y1 + 24}" fill="{P["red"]}" font-size="11">'
                 f'red dashed = OLS (attenuated)</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'OLS minimizes vertical gaps and flattens; TLS minimizes perpendicular gaps and '
                 f'holds the true slope.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
