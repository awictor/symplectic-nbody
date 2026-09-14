"""Theil-Sen demo: a line fit that ignores outliers, next to OLS getting dragged off by them."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import theil_sen as ts


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def uniform(self, a, b):
        return a + (b - a) * self.u()


def main():
    lines = []
    lines.append("Theil-Sen estimator -- robust line fitting by median of pairwise slopes")
    lines.append("=" * 72)
    lines.append("")

    m_true, b_true = 2.0, 3.0
    n = 60
    rng = _R(20)
    xs = [float(i) for i in range(n)]
    clean = [m_true * x + b_true + rng.uniform(-3, 3) for x in xs]
    ys = list(clean)
    # inject 25% gross outliers
    out_idx = []
    n_out = n // 4
    for k in range(n_out):
        idx = (k * 7 + 3) % n
        ys[idx] = clean[idx] + rng.uniform(60, 120)
        out_idx.append(idx)

    m_ts, b_ts = ts.theil_sen(xs, ys)
    m_sg, b_sg = ts.siegel_repeated_median(xs, ys)
    m_ols, b_ols = ts.ols(xs, ys)
    lo, hi = ts.slope_confidence_interval(xs, ys)

    lines.append(f"True line: y = {m_true} x + {b_true}.  {n} points, {n_out} ({100*n_out//n}%) gross outliers.")
    lines.append("")
    lines.append("   method                 slope     intercept   |slope error|")
    lines.append("   " + "-" * 56)
    lines.append(f"   OLS (least squares)   {m_ols:7.3f}   {b_ols:8.3f}     {abs(m_ols - m_true):.3f}")
    lines.append(f"   Theil-Sen             {m_ts:7.3f}   {b_ts:8.3f}     {abs(m_ts - m_true):.3f}")
    lines.append(f"   Siegel repeated-med   {m_sg:7.3f}   {b_sg:8.3f}     {abs(m_sg - m_true):.3f}")
    lines.append("")
    lines.append(f"Theil-Sen 95% slope CI: [{lo:.3f}, {hi:.3f}]  (brackets true slope {m_true})")
    lines.append("")
    lines.append(f"OLS is off by {abs(m_ols - m_true):.2f}; Theil-Sen by {abs(m_ts - m_true):.3f}.")
    lines.append("The median of pairwise slopes tolerates up to 29% bad data; OLS tolerates none.")
    lines.append("")

    # breakdown sweep: how each method degrades as outlier fraction rises
    lines.append("Slope error vs outlier fraction:")
    lines.append("   frac    OLS-err   TheilSen-err   Siegel-err")
    lines.append("   " + "-" * 46)
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45):
        rng2 = _R(5)
        yy = [m_true * x + b_true + rng2.uniform(-2, 2) for x in xs]
        k_out = int(n * frac)
        for k in range(k_out):
            yy[(k * 3) % n] += 300.0
        eo = abs(ts.ols(xs, yy)[0] - m_true)
        et = abs(ts.theil_sen(xs, yy)[0] - m_true)
        es = abs(ts.siegel_repeated_median(xs, yy)[0] - m_true)
        lines.append(f"   {frac:4.2f}   {eo:7.2f}    {et:8.3f}     {es:8.3f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(xs, ys, out_idx, (m_ts, b_ts), (m_ols, b_ols), (m_true, b_true))
    return text, svg


def _svg(xs, ys, out_idx, ts_model, ols_model, true_model):
    W, H = 640, 440
    P = PALETTE
    # frame excludes the far-flung outliers from the y-range for readability, but still plots them at the top
    n = len(xs)
    inlier_y = [ys[i] for i in range(n) if i not in out_idx]
    ymin = min(inlier_y)
    ymax = max(inlier_y)
    pad = 0.15 * (ymax - ymin)
    ymin -= pad
    ymax += pad * 3   # headroom for a few clipped outliers
    xmin, xmax = min(xs), max(xs)

    x0, x1, y0, y1 = 50, 615, 55, 370

    def px(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def py(v):
        v = max(min(v, ymax), ymin)
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Theil-Sen vs OLS with 25% gross outliers</text>')

    out_set = set(out_idx)
    # inlier points (gray) and outliers (red, clipped to top)
    for i in range(n):
        if i in out_set:
            parts.append(f'<circle cx="{px(xs[i]):.1f}" cy="{py(ys[i]):.1f}" r="3" '
                         f'fill="{P["red"]}"/>')
        else:
            parts.append(f'<circle cx="{px(xs[i]):.1f}" cy="{py(ys[i]):.1f}" r="2.4" '
                         f'fill="{P["gray"]}"/>')

    def line(model, color, width, dash=None):
        m, b = model
        y_a = m * xmin + b
        y_bb = m * xmax + b
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<line x1="{px(xmin):.1f}" y1="{py(y_a):.1f}" x2="{px(xmax):.1f}" '
                f'y2="{py(y_bb):.1f}" stroke="{color}" stroke-width="{width}"{d}/>')

    # true line (green), Theil-Sen (yellow, overlays it), OLS (red dashed, pulled up)
    parts.append(line(true_model, P["green"], 3))
    parts.append(line(ts_model, P["yellow"], 2))
    parts.append(line(ols_model, P["red"], 1.8, dash="6,4"))

    parts.append(f'<text x="{x0}" y="{y1 + 24}" fill="{P["green"]}" font-size="11">'
                 f'green = true line</text>')
    parts.append(f'<text x="{x0 + 120}" y="{y1 + 24}" fill="{P["yellow"]}" font-size="11">'
                 f'yellow = Theil-Sen (on truth)</text>')
    parts.append(f'<text x="{x0 + 330}" y="{y1 + 24}" fill="{P["red"]}" font-size="11">'
                 f'red dashed = OLS (dragged up)</text>')
    parts.append(f'<text x="{x0}" y="{y1 + 42}" fill="{P["red"]}" font-size="11">'
                 f'red dots = outliers</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'The median of all pairwise slopes ignores the outliers; least squares chases them.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
