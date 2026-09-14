"""Mann-Kendall demo: detect a monotonic trend in noisy skewed data and quantify it with the Sen slope."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import mann_kendall_trend as mk


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


def ols_slope(x):
    n = len(x)
    t = list(range(n))
    mt, mx = sum(t) / n, sum(x) / n
    stt = sum((ti - mt) ** 2 for ti in t)
    stx = sum((t[i] - mt) * (x[i] - mx) for i in range(n))
    return stx / stt


def main():
    lines = []
    lines.append("Mann-Kendall trend test -- nonparametric monotonic-trend detection")
    lines.append("=" * 66)
    lines.append("")

    # skewed data (exponential-ish noise) with a real upward trend + an outlier
    rng = _R(4)
    n = 50
    x = [0.3 * i + abs(rng.normal()) ** 2 for i in range(n)]   # skewed positive noise
    x[10] = 60.0   # outlier
    res = mk.mann_kendall(x)

    lines.append(f"{n} points, upward trend + skewed noise + one outlier at t=10:")
    lines.append(f"  Mann-Kendall S = {res['S']}, z = {res['z']:.2f}, p = {res['p_value']:.2e}")
    lines.append(f"  tau (effect size) = {res['tau']:.3f}")
    lines.append(f"  trend: {res['trend']}")
    lines.append(f"  Sen slope = {res['sen_slope']:.3f} / step  (robust)")
    lines.append(f"  OLS slope = {ols_slope(x):.3f} / step  (pulled by the outlier + skew)")
    lines.append("")

    # trend strength vs slope
    lines.append("Detection vs true slope (fixed noise):")
    lines.append("   slope    S       p-value      Sen slope")
    lines.append("   " + "-" * 42)
    for sl in (0.0, 0.05, 0.1, 0.3, 0.6):
        rng2 = _R(9)
        xx = [sl * i + rng2.normal() for i in range(40)]
        r = mk.mann_kendall(xx)
        lines.append(f"   {sl:5.2f}   {r['S']:5d}   {r['p_value']:.3e}   {r['sen_slope']:.3f}")
    lines.append("")
    lines.append("The test flags a trend only when S clears the noise band; Sen slope estimates its rate.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(x, res)
    return text, svg


def _svg(x, res):
    W, H = 640, 420
    P = PALETTE
    n = len(x)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Mann-Kendall trend (p={res["p_value"]:.1e}) with the robust Sen slope</text>')

    x0, x1, y0, y1 = 50, 615, 55, 360
    vmin, vmax = min(x), max(x)
    pad = 0.08 * (vmax - vmin)
    vmin -= pad
    vmax += pad

    def px(i):
        return x0 + i / (n - 1) * (x1 - x0)

    def py(v):
        return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)

    # data points
    for i in range(n):
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(x[i]):.1f}" r="3" fill="{P["blue"]}" opacity="0.75"/>')

    # Sen slope line through the median intercept
    import theil_sen as ts
    slope, intercept = ts.theil_sen(list(range(n)), x)
    ya = intercept
    yb = intercept + slope * (n - 1)
    parts.append(f'<line x1="{px(0):.1f}" y1="{py(ya):.1f}" x2="{px(n-1):.1f}" y2="{py(yb):.1f}" '
                 f'stroke="{P["yellow"]}" stroke-width="2.5"/>')

    # OLS line for comparison
    def ols_line(x):
        nn = len(x)
        t = list(range(nn))
        mt, mx = sum(t)/nn, sum(x)/nn
        stt = sum((ti-mt)**2 for ti in t)
        stx = sum((t[i]-mt)*(x[i]-mx) for i in range(nn))
        sl = stx/stt
        return mx - sl*mt, sl
    b0, sl = ols_line(x)
    parts.append(f'<line x1="{px(0):.1f}" y1="{py(b0):.1f}" x2="{px(n-1):.1f}" y2="{py(b0 + sl*(n-1)):.1f}" '
                 f'stroke="{P["red"]}" stroke-width="1.5" stroke-dasharray="6,4"/>')

    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{x0}" y="{y1 + 22:.1f}" fill="{P["blue"]}" font-size="11">blue = data (skewed + outlier)</text>')
    parts.append(f'<text x="{x0 + 250}" y="{y1 + 22:.1f}" fill="{P["yellow"]}" font-size="11">yellow = Sen slope (robust)</text>')
    parts.append(f'<text x="{x0}" y="{y1 + 40:.1f}" fill="{P["red"]}" font-size="11">red dashed = OLS (pulled by outlier)</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'Mann-Kendall counts pairwise up/down signs -- a monotone trend clears the noise band '
                 f'regardless of shape or outliers.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
