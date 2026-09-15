"""Moran process demo: fixation probability vs selection, and single-mutant fixation approaching 1-1/r."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import moran_process as mp


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Moran process -- finite-population evolution, one birth & death per step")
    lines.append("=" * 72)
    lines.append("")

    n = 12
    lines.append(f"Population N={n}. Fixation probability of type-A from i copies, fitness ratio r:")
    lines.append("   i    r=0.5 (del)   r=1.0 (neutral)   r=2.0 (adv)")
    lines.append("   " + "-" * 48)
    for i in (1, 3, 6, 9, 11):
        d = mp.fixation_probability_formula(n, i, 0.5)
        neu = mp.fixation_probability_formula(n, i, 1.0)
        adv = mp.fixation_probability_formula(n, i, 2.0)
        lines.append(f"   {i:2d}    {d:.4f}        {neu:.4f}            {adv:.4f}")
    lines.append("")
    lines.append("Neutral fixation = i/N exactly; advantage bends the curve up, disadvantage down.")
    lines.append("")

    # single mutant fixation vs 1 - 1/r
    lines.append("A single advantageous mutant (i=1) fixes with prob (1-1/r)/(1-1/r^N):")
    lines.append("   r      N=12     N=100    N=1000    limit 1-1/r")
    lines.append("   " + "-" * 46)
    for r in (1.1, 1.5, 2.0, 3.0):
        f12 = mp.single_mutant_fixation(12, r)
        f100 = mp.single_mutant_fixation(100, r)
        f1000 = mp.single_mutant_fixation(1000, r)
        lines.append(f"   {r:4.1f}   {f12:.4f}   {f100:.4f}   {f1000:.4f}    {1 - 1/r:.4f}")
    lines.append("")
    lines.append("Even a strongly-beneficial mutant is far from guaranteed to fix -- drift can lose it early.")
    lines.append("")

    # empirical validation
    r = 1.5
    i0 = 3
    theo = mp.fixation_probability_formula(n, i0, r)
    emp = mp.fixation_probability(n, i0, r, n_runs=5000, seed=1)
    lines.append(f"Validation (N={n}, i0={i0}, r={r}): formula {theo:.3f}, "
                 f"empirical {emp:.3f} over 5000 runs.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(n)
    return text, svg


def _svg(n):
    W, H = 640, 420
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Moran fixation probability vs starting count, by selection</text>')

    x0, x1, y0, y1 = 55, 610, 55, 350

    def px(i):
        return x0 + i / n * (x1 - x0)

    def py(p):
        return y1 - p * (y1 - y0)

    # diagonal (neutral = i/N)
    curves = [(0.5, P["red"], "r=0.5 deleterious"),
              (1.0, P["gray"], "r=1.0 neutral (=i/N)"),
              (1.5, P["yellow"], "r=1.5"),
              (3.0, P["green"], "r=3.0 advantageous")]
    for r, col, label in curves:
        pts = " ".join(f"{px(i):.1f},{py(mp.fixation_probability_formula(n, i, r)):.1f}" for i in range(n + 1))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')

    # axes
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{x0 - 8}" y="{py(1.0) + 4:.1f}" fill="{P["gray"]}" font-size="10" text-anchor="end">1</text>')
    parts.append(f'<text x="{x0 - 8}" y="{py(0.0) + 4:.1f}" fill="{P["gray"]}" font-size="10" text-anchor="end">0</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1 + 20:.1f}" fill="{P["gray"]}" '
                 f'font-size="11" text-anchor="middle">starting count i (of N={n})</text>')

    # legend
    ly = y0 + 10
    for r, col, label in curves:
        parts.append(f'<text x="{x0 + 20}" y="{ly:.1f}" fill="{col}" font-size="10">{label}</text>')
        ly += 15

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'the neutral line is the diagonal i/N; selection curves it -- advantage lifts small-i '
                 f'fixation, disadvantage crushes it.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
