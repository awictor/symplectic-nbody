"""Demo: belief propagation (sum-product) -- exact marginals on a tree factor graph.

Builds a small tree-structured Markov random field (a chain of binary variables with a coupling that
prefers agreement, plus a biased end), computes every marginal by message passing, and verifies it
matches brute-force enumeration. Draws the factor graph and the marginal bars.

    python examples/belief_propagation_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from belief_propagation import FactorGraph, sum_product, max_product, brute_marginals  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Belief propagation (sum-product): exact marginals on a tree, by message passing\n")

    # a 5-variable chain of binary spins; pairwise factors prefer neighbours to agree,
    # and one end is biased toward state 1.
    n = 5
    variables = {f"s{i}": 2 for i in range(n)}
    agree = [[3.0, 1.0], [1.0, 3.0]]  # prefer equal neighbours
    factors = []
    factors.append((("s0",), [1.0, 4.0]))  # s0 biased toward 1
    for i in range(n - 1):
        factors.append(((f"s{i}", f"s{i+1}"), agree))
    fg = FactorGraph(variables, factors)

    print(f"  A chain of {n} binary spins; neighbours prefer to agree (factor [[3,1],[1,3]]),")
    print("  and s0 is biased toward state 1 (unary [1,4]).\n")
    print(f"  Factor graph is a tree: {fg.is_tree()}\n")

    marg, logZ = sum_product(fg)
    bmarg, _ = brute_marginals(fg)
    print(f"  {'variable':>9}  {'P(state=1) BP':>14}  {'brute':>8}")
    for i in range(n):
        v = f"s{i}"
        print(f"  {v:>9}  {marg[v][1]:>14.4f}  {bmarg[v][1]:>8.4f}")
    print("\n  The bias at s0 propagates down the chain, decaying with distance -- BP captures it")
    print("  exactly, matching brute enumeration, in time linear in the graph.")

    import math
    print(f"\n  Partition function Z = {math.exp(logZ):.3f}")

    mp, _ = max_product(fg)
    print(f"  MAP (most probable) configuration: {[mp[f's{i}'] for i in range(n)]}")

    _svg(os.path.join(outdir, "belief_propagation.svg"), fg, marg, n)
    print(f"\n  wrote {os.path.join(outdir, 'belief_propagation.svg')}")


def _svg(path, fg, marg, n, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="28" fill="#e6edf3" font-size="15">'
        'Chain factor graph (top) and BP marginals P(state=1) (bottom)</text>',
    ]
    # factor graph: variables as circles, factors as small squares between them
    y = 110
    x0 = 80
    gap = (width - 160) / (n - 1)
    vpos = {i: (x0 + i * gap, y) for i in range(n)}
    # pairwise factor squares between neighbours
    for i in range(n - 1):
        mx = (vpos[i][0] + vpos[i + 1][0]) / 2
        parts.append(f'<line x1="{vpos[i][0]:.0f}" y1="{y}" x2="{mx:.0f}" y2="{y}" '
                     f'stroke="#30363d" stroke-width="1.5"/>')
        parts.append(f'<line x1="{mx:.0f}" y1="{y}" x2="{vpos[i+1][0]:.0f}" y2="{y}" '
                     f'stroke="#30363d" stroke-width="1.5"/>')
        parts.append(f'<rect x="{mx-7:.0f}" y="{y-7}" width="14" height="14" fill="#b197fc"/>')
    # unary factor on s0
    parts.append(f'<rect x="{vpos[0][0]-7:.0f}" y="{y-42}" width="14" height="14" fill="#ff922b"/>')
    parts.append(f'<line x1="{vpos[0][0]:.0f}" y1="{y-28}" x2="{vpos[0][0]:.0f}" y2="{y-8}" '
                 f'stroke="#30363d" stroke-width="1.5"/>')
    for i in range(n):
        x, yy = vpos[i]
        parts.append(f'<circle cx="{x:.0f}" cy="{yy}" r="15" fill="#161b22" stroke="#4dabf7" '
                     f'stroke-width="2"/>')
        parts.append(f'<text x="{x:.0f}" y="{yy+4}" fill="#4dabf7" font-size="10" '
                     f'text-anchor="middle">s{i}</text>')

    # marginal bars
    by0 = 220
    bh = 130
    parts.append(f'<line x1="{x0-20}" y1="{by0+bh}" x2="{width-40}" y2="{by0+bh}" '
                 f'stroke="#8b949e" stroke-width="1"/>')
    parts.append(f'<line x1="{x0-20}" y1="{by0+bh/2:.0f}" x2="{width-40}" y2="{by0+bh/2:.0f}" '
                 f'stroke="#30363d" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{x0-24}" y="{by0+bh/2+3:.0f}" fill="#8b949e" font-size="8" '
                 f'text-anchor="end">0.5</text>')
    bw = 40
    for i in range(n):
        p = marg[f"s{i}"][1]
        h = bh * p
        x = vpos[i][0] - bw / 2
        parts.append(f'<rect x="{x:.0f}" y="{by0+bh-h:.0f}" width="{bw}" height="{h:.0f}" '
                     f'fill="#06d6a0"/>')
        parts.append(f'<text x="{vpos[i][0]:.0f}" y="{by0+bh-h-5:.0f}" fill="#e6edf3" '
                     f'font-size="9" text-anchor="middle">{p:.2f}</text>')
        parts.append(f'<text x="{vpos[i][0]:.0f}" y="{by0+bh+16:.0f}" fill="#8b949e" '
                     f'font-size="9" text-anchor="middle">s{i}</text>')
    parts.append(f'<text x="{x0-20}" y="{by0-8}" fill="#8b949e" font-size="10">'
                 f'P(spin = 1): the s0 bias decays down the chain</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
