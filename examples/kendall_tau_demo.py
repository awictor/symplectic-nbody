"""Kendall tau demo: rank correlation robust to outliers and nonlinearity where Pearson r fails."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import kendall_tau as kt


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


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0


def main():
    lines = []
    lines.append("Kendall tau -- rank correlation from concordant vs discordant pairs")
    lines.append("=" * 68)
    lines.append("")

    # monotone but curved relationship: Pearson underestimates, tau sees full association
    rng = _R(4)
    x = [i / 3.0 for i in range(1, 26)]
    y = [math.exp(v / 3) + 0.15 * rng.normal() for v in x]   # increasing, curved
    res = kt.kendall(x, y)
    lines.append("Monotone but CURVED relationship (y ~ exp):")
    lines.append(f"  Kendall tau-b = {res['tau_b']:.3f}  (p = {res['p_value']:.2e})")
    lines.append(f"  Pearson r     = {pearson(x, y):.3f}   (linear measure, underrates the monotone link)")
    lines.append("")

    # outlier robustness: one wild point wrecks Pearson, barely moves tau
    x2 = list(range(20))
    y2 = [v + rng.normal() for v in x2]   # strong linear
    r_clean = pearson(x2, y2)
    t_clean = kt.tau_b(x2, y2)
    x2o = x2 + [10]
    y2o = y2 + [500]   # a wild outlier off the trend
    r_out = pearson(x2o, y2o)
    t_out = kt.tau_b(x2o, y2o)
    lines.append("Outlier robustness (add one wild point to a clean linear trend):")
    lines.append(f"  Pearson r : {r_clean:.3f} -> {r_out:.3f}  (crashes)")
    lines.append(f"  Kendall t : {t_clean:.3f} -> {t_out:.3f}  (barely moves -- ranks ignore magnitude)")
    lines.append("")

    # concordant/discordant breakdown
    c, d, xt, yt = kt._counts(x, y)
    lines.append(f"Pair breakdown for the curved data ({len(x)} points, {res['n_pairs']} pairs):")
    lines.append(f"  concordant = {c}, discordant = {d}, x-ties = {xt}, y-ties = {yt}")
    lines.append(f"  tau = (C - D) / pairs = ({c} - {d}) / {res['n_pairs']} = {res['tau_a']:.3f}")
    lines.append("")

    # tau vs association strength
    lines.append("tau tracks the fraction of concordant pairs:")
    lines.append("   noise    tau_b    P(concordant)")
    lines.append("   " + "-" * 34)
    base = list(range(40))
    for noise in (0.0, 5.0, 15.0, 40.0, 100.0):
        rng2 = _R(2)
        yy = [v + noise * rng2.normal() for v in base]
        t = kt.tau_b(base, yy)
        p_conc = (t + 1) / 2   # tau = 2 P(conc) - 1 for no ties
        lines.append(f"   {noise:5.0f}    {t:+.3f}     {p_conc:.3f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(x, y, res)
    return text, svg


def _svg(x, y, res):
    W, H = 640, 430
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Concordant (green) vs discordant (red) pairs; tau-b = {res["tau_b"]:.3f}</text>')

    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    padx = 0.08 * (xmax - xmin)
    pady = 0.08 * (ymax - ymin)
    x0, x1, y0, y1 = 55, 610, 55, 370

    def px(v):
        return x0 + (v - xmin + padx) / (xmax - xmin + 2 * padx) * (x1 - x0)

    def py(v):
        return y1 - (v - ymin + pady) / (ymax - ymin + 2 * pady) * (y1 - y0)

    n = len(x)
    # draw a sample of pairs as faint concordant/discordant links (subsample to avoid clutter)
    step = max(1, (n * (n - 1) // 2) // 120)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            k += 1
            if k % step != 0:
                continue
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s == 0:
                continue
            col = P["green"] if s > 0 else P["red"]
            parts.append(f'<line x1="{px(x[i]):.1f}" y1="{py(y[i]):.1f}" '
                         f'x2="{px(x[j]):.1f}" y2="{py(y[j]):.1f}" '
                         f'stroke="{col}" stroke-width="0.5" opacity="0.25"/>')

    # points on top
    for i in range(n):
        parts.append(f'<circle cx="{px(x[i]):.1f}" cy="{py(y[i]):.1f}" r="4" '
                     f'fill="{P["blue"]}"/>')

    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="20" y="{H - 26}" fill="{P["gray"]}" font-size="11">'
                 f'blue dots = data; green links join concordant pairs (same ordering), red join '
                 f'discordant pairs.</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'a monotone cloud is almost all green -> tau near +1, whatever the curve shape.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
