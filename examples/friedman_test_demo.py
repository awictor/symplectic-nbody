"""Friedman demo: rank treatments within blocks to find a treatment effect ANOVA would miss under block noise."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import friedman_test as ft


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}
COLORS = ["#4dabf7", "#ffd43b", "#06d6a0", "#ff6b6b"]


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
    lines.append("Friedman test -- nonparametric repeated-measures ANOVA")
    lines.append("=" * 54)
    lines.append("")

    # 4 algorithms scored on 15 datasets; each dataset has its own difficulty (block effect)
    names = ["algo-A", "algo-B", "algo-C", "algo-D"]
    true_effect = [0.0, 1.0, 2.0, 3.5]
    rng = _R(4)
    n_blocks = 15
    data = []
    for _ in range(n_blocks):
        difficulty = 10 * rng.normal()   # huge per-dataset offset (block effect)
        data.append([difficulty + true_effect[j] + 0.5 * rng.normal() for j in range(4)])

    res = ft.friedman(data)
    lines.append(f"{n_blocks} datasets x 4 algorithms. Each dataset has a large difficulty offset")
    lines.append("(a block effect ~10x the treatment differences).")
    lines.append("")
    lines.append(f"Friedman Q = {res['Q']:.2f} (df={res['df']}), p = {res['p_value']:.3e}")
    lines.append(f"Kendall's W (concordance) = {res['kendall_w']:.3f}")
    lines.append("")
    lines.append("Average rank per algorithm (lower = better score more often):")
    for j, nm in enumerate(names):
        lines.append(f"  {nm}: avg rank {res['avg_ranks'][j]:.2f}")
    lines.append("")
    lines.append("Ranking WITHIN each dataset cancels the difficulty offset, exposing the algorithm")
    lines.append("effect a raw-score ANOVA would drown in block-to-block variance.")
    lines.append("")

    # show block-invariance
    shifted = [[data[i][j] + 1000 * i for j in range(4)] for i in range(n_blocks)]
    res_shift = ft.friedman(shifted)
    lines.append(f"Add an arbitrary per-block constant: Q unchanged ({res['Q']:.2f} -> {res_shift['Q']:.2f}).")
    lines.append("")

    # power vs effect size
    lines.append("Power vs treatment spacing (15 blocks, block noise 10x):")
    lines.append("   spacing   Q        p-value")
    lines.append("   " + "-" * 32)
    for spacing in (0.0, 0.5, 1.0, 2.0, 4.0):
        rng2 = _R(9)
        d = []
        for _ in range(15):
            diff = 10 * rng2.normal()
            d.append([diff + spacing * j + 0.5 * rng2.normal() for j in range(4)])
        r = ft.friedman(d)
        lines.append(f"   {spacing:6.1f}   {r['Q']:6.2f}   {r['p_value']:.3e}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(data, names, res)
    return text, svg


def _svg(data, names, res):
    W, H = 640, 450
    P = PALETTE
    n = len(data)
    k = len(names)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Within-block ranks per algorithm (Q={res["Q"]:.1f}, p={res["p_value"]:.1e})</text>')

    ranks = ft._rank_within_blocks(data)

    # TOP: rank of each algorithm in each block (jittered dots by treatment color)
    tx0, tx1, ty0, ty1 = 60, 610, 55, 250

    def bx(i):
        return tx0 + i / (n - 1) * (tx1 - tx0)

    def ry(rank):
        # rank 1 (best) at top, k at bottom
        return ty0 + (rank - 1) / (k - 1) * (ty1 - ty0)

    for j in range(k):
        col = COLORS[j % len(COLORS)]
        pts = " ".join(f"{bx(i):.1f},{ry(ranks[i][j]):.1f}" for i in range(n))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1" opacity="0.5"/>')
        for i in range(n):
            parts.append(f'<circle cx="{bx(i):.1f}" cy="{ry(ranks[i][j]):.1f}" r="3" fill="{col}"/>')
    for rk in range(1, k + 1):
        parts.append(f'<text x="{tx0 - 20:.1f}" y="{ry(rk) + 4:.1f}" fill="{P["gray"]}" '
                     f'font-size="10">r{rk}</text>')
    parts.append(f'<text x="{tx0}" y="{ty1 + 16:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'each colored track = one algorithm across the 15 datasets (within-block rank)</text>')

    # BOTTOM: average rank bars per algorithm
    bx0, by0, by1 = 60, 300, 400
    bw = (tx1 - bx0) / k
    for j in range(k):
        col = COLORS[j % len(COLORS)]
        ar = res["avg_ranks"][j]
        h = (ar / k) * (by1 - by0)
        x = bx0 + j * bw + bw * 0.2
        parts.append(f'<rect x="{x:.1f}" y="{by1 - h:.1f}" width="{bw*0.6:.1f}" height="{h:.1f}" '
                     f'fill="{col}"/>')
        parts.append(f'<text x="{x + bw*0.3:.1f}" y="{by1 + 14:.1f}" fill="{P["text"]}" '
                     f'font-size="10" text-anchor="middle">{names[j]}</text>')
        parts.append(f'<text x="{x + bw*0.3:.1f}" y="{by1 - h - 4:.1f}" fill="{col}" '
                     f'font-size="10" text-anchor="middle">{ar:.2f}</text>')
    parts.append(f'<text x="{bx0}" y="{by0 - 6:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'average rank per algorithm -- the spread is what Q measures</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'ranking within each dataset removes its difficulty offset; the average-rank spread '
                 f'reveals the treatment effect.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
