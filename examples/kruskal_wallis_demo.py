"""Kruskal-Wallis demo: nonparametric ANOVA across several groups, with the rank distribution and chi2 null."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import kruskal_wallis as kw


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}
COLORS = ["#4dabf7", "#ffd43b", "#06d6a0", "#b197fc", "#ff6b6b"]


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
    lines.append("Kruskal-Wallis H test -- nonparametric one-way ANOVA")
    lines.append("=" * 54)
    lines.append("")

    # three treatment groups, one clearly elevated, all skewed (non-normal) with an outlier
    rng = _R(7)
    groups = {
        "control": [abs(rng.normal()) * 2 + 3 for _ in range(18)],
        "drug-lo": [abs(rng.normal()) * 2 + 4 for _ in range(18)],
        "drug-hi": [abs(rng.normal()) * 2 + 7 for _ in range(18)],
    }
    # inject an outlier into control (ANOVA-hostile, KW-robust)
    groups["control"][0] = 40.0

    names = list(groups)
    data = [groups[n] for n in names]
    res = kw.kruskal_wallis(data)

    lines.append("Three skewed groups (18 each), control has an outlier of 40:")
    for i, n in enumerate(names):
        g = groups[n]
        lines.append(f"  {n:8s}: median {sorted(g)[len(g)//2]:.2f}, mean rank {res['mean_ranks'][i]:.1f}")
    lines.append("")
    lines.append(f"H = {res['H']:.3f}  (df={res['df']})")
    lines.append(f"p-value = {res['p_value']:.3e}  (chi-squared approximation)")
    lines.append(f"epsilon-squared effect size = {res['epsilon_squared']:.3f}")
    lines.append("")
    lines.append("The rank-based test ignores the outlier's magnitude -- it counts only positions.")
    lines.append("")

    # power vs separation
    lines.append("H and p as the groups separate (shift applied to drug-hi):")
    lines.append("   shift    H        p-value")
    lines.append("   " + "-" * 30)
    base = [rng.normal() for _ in range(20)]
    mid = [rng.normal() + 1 for _ in range(20)]
    for shift in (0.0, 1.0, 2.0, 4.0, 8.0):
        hi = [rng.normal() + shift for _ in range(20)]
        r = kw.kruskal_wallis([base, mid, hi])
        lines.append(f"   {shift:5.1f}    {r['H']:6.2f}   {r['p_value']:.3e}")
    lines.append("")

    # chi-squared critical values used for the p-value
    lines.append("Chi-squared survival function (the null for H), df=2:")
    for x in (1.0, 2.0, 4.0, 5.991, 9.21):
        lines.append(f"   P(chi2_2 > {x:5.3f}) = {kw.chi2_sf(x, 2):.4f}")
    lines.append("   (5.991 is the 0.05 critical value, 9.21 the 0.01)")

    text = "\n".join(lines)
    print(text)

    svg = _svg(names, data, res)
    return text, svg


def _svg(names, data, res):
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Kruskal-Wallis: pooled ranks per group (H={res["H"]:.2f}, p={res["p_value"]:.1e})</text>')

    # rank each observation, plot rank position per group as a strip
    from mann_whitney import _average_ranks
    all_vals = [v for g in data for v in g]
    ranks = _average_ranks(all_vals)
    N = len(all_vals)

    x0, x1 = 70, 610
    top, bot = 60, 330
    band = (bot - top) / len(names)

    def rx(r):
        return x0 + (r - 1) / (N - 1) * (x1 - x0)

    idx = 0
    for gi, name in enumerate(names):
        y = top + gi * band + band / 2
        col = COLORS[gi % len(COLORS)]
        parts.append(f'<text x="10" y="{y + 4:.1f}" fill="{col}" font-size="11">{name}</text>')
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#30363d" stroke-width="1"/>')
        n_g = len(data[gi])
        r = ranks[idx:idx + n_g]
        idx += n_g
        for rr in r:
            parts.append(f'<circle cx="{rx(rr):.1f}" cy="{y:.1f}" r="4" fill="{col}" opacity="0.75"/>')
        # mean-rank marker
        mr = res["mean_ranks"][gi]
        parts.append(f'<line x1="{rx(mr):.1f}" y1="{y - band/2 + 6:.1f}" x2="{rx(mr):.1f}" '
                     f'y2="{y + band/2 - 6:.1f}" stroke="{P["text"]}" stroke-width="2"/>')

    # grand mean rank
    grand = (N + 1) / 2
    parts.append(f'<line x1="{rx(grand):.1f}" y1="{top}" x2="{rx(grand):.1f}" y2="{bot}" '
                 f'stroke="{P["gray"]}" stroke-width="1.5" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{rx(grand):.1f}" y="{bot + 16:.1f}" fill="{P["gray"]}" '
                 f'font-size="10" text-anchor="middle">grand mean rank</text>')

    parts.append(f'<text x="20" y="{bot + 40:.1f}" fill="{P["gray"]}" font-size="11">'
                 f'each dot is one observation at its pooled rank; white bars = group mean ranks.</text>')
    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'H measures how far the group mean ranks spread from the grand mean -- large spread, '
                 f'small p.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
