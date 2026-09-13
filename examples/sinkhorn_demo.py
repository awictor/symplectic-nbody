"""Demo: Sinkhorn optimal transport -- morphing one distribution into another at least cost.

Transports one histogram to another, shows the transport plan and how the regularized cost approaches
the exact Wasserstein distance as the entropy penalty shrinks, and draws the two distributions with
the transport plan as a heatmap.

    python examples/sinkhorn_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sinkhorn import (  # noqa: E402
    sinkhorn,
    cost_matrix,
    exact_transport_1d,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sinkhorn optimal transport: least-cost plan to reshape one distribution into another\n")

    # two bimodal-ish histograms over the same support
    xs = [float(i) for i in range(10)]
    a = [0.02, 0.05, 0.15, 0.25, 0.20, 0.12, 0.10, 0.06, 0.03, 0.02]
    b = [0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.22, 0.18, 0.10, 0.06]
    a = [v / sum(a) for v in a]
    b = [v / sum(b) for v in b]
    C = cost_matrix(xs, xs, p=2)

    exact = exact_transport_1d(a, xs, b, xs, p=2)
    print(f"  Exact 1-D optimal transport cost (squared): {exact:.4f}\n")

    print(f"  {'eps':>8}  {'iterations':>11}  {'regularized cost':>17}")
    for eps in [1.0, 0.5, 0.2, 0.1, 0.05]:
        P, cost, iters = sinkhorn(a, b, C, eps=eps, max_iter=20000)
        print(f"  {eps:>8.2f}  {iters:>11}  {cost:>17.4f}")
    print(f"    -> as the entropy penalty eps shrinks, the cost approaches the exact {exact:.4f}.\n")

    P, cost, _ = sinkhorn(a, b, C, eps=0.05, max_iter=20000)
    print("  A slice of the transport plan (rows = source bins, cols = target bins, x1000):")
    print("      " + "".join(f"{j:>4}" for j in range(10)))
    for i in range(10):
        row = "".join(f"{int(round(P[i][j] * 1000)):>4}" for j in range(10))
        print(f"    {i:>2}  {row}")
    print("\n  Mass moves from the left peak of 'a' toward the right peak of 'b'; the plan sits near")
    print("  the diagonal because moving mass a short distance is cheap. The total cost is the")
    print("  Wasserstein (earth mover's) distance -- a geometry-aware metric between distributions.")

    _svg(os.path.join(outdir, "sinkhorn.svg"), xs, a, b, P)
    print(f"\n  wrote {os.path.join(outdir, 'sinkhorn.svg')}")


def _svg(path, xs, a, b, P, width=760, height=430):
    n = len(xs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Source a (blue) and target b (green) histograms; transport plan heatmap (right)</text>',
    ]

    # ---- left: two histograms -----------------------------------------------------------
    ox, oy, ow, oh = 40, 60, 300, 300
    bw = ow / n
    amax = max(max(a), max(b))
    parts.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="#8b949e"/>')
    for i in range(n):
        ha = oh / 2 * a[i] / amax
        hb = oh / 2 * b[i] / amax
        x = ox + i * bw
        parts.append(f'<rect x="{x+2:.1f}" y="{oy+oh-ha:.1f}" width="{bw/2-2:.1f}" '
                     f'height="{ha:.1f}" fill="#4dabf7"/>')
        parts.append(f'<rect x="{x+bw/2:.1f}" y="{oy+oh-hb:.1f}" width="{bw/2-2:.1f}" '
                     f'height="{hb:.1f}" fill="#06d6a0"/>')
    parts.append(f'<text x="{ox}" y="{oy+oh+18}" fill="#8b949e" font-size="10">'
                 f'blue = source a, green = target b</text>')

    # ---- right: transport plan heatmap --------------------------------------------------
    gx, gy = 430, 60
    cell = min((width - gx - 30) // n, oh // n)
    pmax = max(max(row) for row in P)
    for i in range(n):
        for j in range(n):
            t = P[i][j] / pmax if pmax > 0 else 0
            r = int(0x0d + t * (0xff - 0x0d))
            g = int(0x11 + t * (0xd4 - 0x11))
            b_ = int(0x17 + t * (0x3b - 0x17))
            parts.append(f'<rect x="{gx + j*cell}" y="{gy + i*cell}" width="{cell-1}" '
                         f'height="{cell-1}" fill="rgb({r},{g},{b_})"/>')
    parts.append(f'<text x="{gx}" y="{gy + n*cell + 18}" fill="#8b949e" font-size="10">'
                 f'plan P (bright = more mass moved); near-diagonal = short moves</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
