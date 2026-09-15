"""Chinese restaurant process demo: clusters emerging without a preset count, and alpha ln n table growth."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import chinese_restaurant as crp


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}
COLORS = ["#4dabf7", "#ffd43b", "#06d6a0", "#b197fc", "#ff6b6b", "#f783ac", "#63e6be", "#ffa94d"]


def main():
    lines = []
    lines.append("Chinese restaurant process -- clustering without a preset number of clusters")
    lines.append("=" * 76)
    lines.append("")

    n = 200
    lines.append(f"{n} customers seated by 'sit at a table with prob n_k/(n-1+alpha), else new table'.")
    lines.append("")
    lines.append("Concentration alpha controls the number of clusters (E[K] = sum alpha/(alpha+i-1)):")
    lines.append("   alpha   E[K] formula   empirical mean   ~ alpha ln n")
    lines.append("   " + "-" * 54)
    for alpha in (0.5, 1.0, 3.0, 10.0):
        theo = crp.expected_tables(n, alpha)
        emp = crp.mean_tables(n, alpha, n_runs=1500, seed=1)
        lines.append(f"   {alpha:5.1f}   {theo:8.2f}       {emp:8.2f}         {alpha * math.log(n):.2f}")
    lines.append("")
    lines.append("Table count grows LOGARITHMICALLY in n -- a few big clusters plus a long tail of small ones.")
    lines.append("")

    # a single realization's cluster sizes
    alpha = 2.0
    sizes = crp.table_size_distribution(n, alpha, seed=7)
    lines.append(f"One realization (alpha={alpha}, n={n}): {len(sizes)} tables, sizes")
    lines.append(f"  {sizes}")
    lines.append(f"  largest {sizes[0]}, {sum(1 for s in sizes if s == 1)} singletons -- the rich-get-richer signature.")
    lines.append("")

    # exchangeability
    lines.append("Exchangeability: partition probability depends only on block SIZES, not order.")
    a1 = [0, 0, 0, 1, 1]
    a2 = [0, 1, 0, 1, 0]
    lines.append(f"  seating {a1} -> {crp.sequential_probability(a1, 1.5):.6f}")
    lines.append(f"  seating {a2} -> {crp.sequential_probability(a2, 1.5):.6f}  (same blocks {{3,2}}, equal prob)")
    lines.append("  -> the CRP is the predictive rule of a Dirichlet process (Bayesian nonparametrics).")

    text = "\n".join(lines)
    print(text)

    svg = _svg(n)
    return text, svg


def _svg(n):
    W, H = 640, 450
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Expected tables vs n (top) and one realization\'s clusters (bottom)</text>')

    # TOP: E[K] vs n for several alpha
    tx0, tx1, ty0, ty1 = 55, 610, 50, 210
    ns = list(range(1, n + 1))
    alphas = [(0.5, P["blue"]), (1.0, P["green"]), (3.0, P["yellow"]), (10.0, P["red"])]
    kmax = max(crp.expected_tables(n, a) for a, _ in alphas)

    def tx(nn):
        return tx0 + nn / n * (tx1 - tx0)

    def ty(k):
        return ty1 - k / kmax * (ty1 - ty0)

    for alpha, col in alphas:
        pts = " ".join(f"{tx(nn):.1f},{ty(crp.expected_tables(nn, alpha)):.1f}" for nn in ns)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.5"/>')
        parts.append(f'<text x="{tx1 + 2:.1f}" y="{ty(crp.expected_tables(n, alpha)) + 4:.1f}" '
                     f'fill="{col}" font-size="9">a={alpha}</text>')
    parts.append(f'<line x1="{tx0}" y1="{ty1}" x2="{tx1}" y2="{ty1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{tx0}" y="{ty0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'E[K] grows like alpha ln n (concave) -- more alpha, more clusters</text>')

    # BOTTOM: cluster sizes of one realization as bars
    alpha = 2.0
    sizes = crp.table_size_distribution(n, alpha, seed=7)
    bx0, bx1, by0, by1 = 55, 610, 260, 410
    bw = (bx1 - bx0) / len(sizes)
    smax = max(sizes)

    def by(s):
        return by1 - s / smax * (by1 - by0)

    for i, s in enumerate(sizes):
        h = by1 - by(s)
        col = COLORS[i % len(COLORS)]
        parts.append(f'<rect x="{bx0 + i * bw + 0.5:.1f}" y="{by(s):.1f}" width="{max(bw-1,1):.1f}" '
                     f'height="{h:.1f}" fill="{col}"/>')
    parts.append(f'<line x1="{bx0}" y1="{by1}" x2="{bx1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{bx0}" y="{by0 - 6:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'cluster sizes (alpha={alpha}, n={n}): a few dominant tables + a long tail of singletons</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'the CRP discovers the cluster count from the data via alpha -- the basis of infinite '
                 f'mixture models.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
