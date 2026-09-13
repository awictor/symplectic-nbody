"""Demo: force-directed graph layout (Fruchterman-Reingold).

Lays out three graphs -- a ring, a two-cluster network, and a small tree -- by simulating springs and
repulsion, reports how the layout energy drops as the system relaxes, and draws all three.

    python examples/force_layout_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from force_layout import (  # noqa: E402
    fruchterman_reingold,
    energy,
    ideal_edge_length,
    mean_edge_length,
    mean_nonedge_length,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Force-directed layout (Fruchterman-Reingold): nodes repel, edges pull\n")

    # graph 1: a ring
    n1 = 10
    ring = [(i, (i + 1) % n1) for i in range(n1)]

    # graph 2: two clusters joined by a bridge
    clusters = []
    for i in range(5):
        for j in range(i + 1, 5):
            clusters.append((i, j))
    for i in range(5, 10):
        for j in range(i + 1, 10):
            clusters.append((i, j))
    clusters.append((4, 5))

    # graph 3: a small binary tree
    tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

    graphs = [("ring (10-cycle)", 10, ring), ("two clusters + bridge", 10, clusters),
              ("binary tree", 7, tree)]

    layouts = []
    for name, n, edges in graphs:
        k = ideal_edge_length(n)
        pos0 = fruchterman_reingold(n, edges, iterations=0, seed=7)  # initial random
        pos = fruchterman_reingold(n, edges, iterations=400, seed=7)
        e0 = energy(pos0, edges, k)
        e1 = energy(pos, edges, k)
        me = mean_edge_length(pos, edges)
        mn = mean_nonedge_length(pos, edges)
        print(f"  {name}:")
        print(f"    energy {e0:8.2f} -> {e1:8.2f}   mean edge {me:.3f}, mean non-edge {mn:.3f}")
        layouts.append((name, n, edges, pos))

    print("\n  Energy drops as the layout relaxes; edges end up shorter than non-edges, so connected")
    print("  nodes cluster and the graph's structure becomes visible without any manual placement.")

    _svg(os.path.join(outdir, "force_layout.svg"), layouts)
    print(f"\n  wrote {os.path.join(outdir, 'force_layout.svg')}")


def _svg(path, layouts, width=760, height=300):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="24" fill="#e6edf3" font-size="15">'
        'Three graphs laid out by simulated springs + repulsion</text>',
    ]
    panel_w = width / len(layouts)
    cols = ["#4dabf7", "#06d6a0", "#ffd43b"]
    for pi, (name, n, edges, pos) in enumerate(layouts):
        # normalize positions into this panel
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        px0 = pi * panel_w + 25
        py0 = 55
        pw = panel_w - 50
        ph = height - 100

        def sx(x):
            return px0 + pw * (x - xmin) / (xmax - xmin or 1)

        def sy(y):
            return py0 + ph * (y - ymin) / (ymax - ymin or 1)

        col = cols[pi % len(cols)]
        for u, v in edges:
            parts.append(f'<line x1="{sx(pos[u][0]):.1f}" y1="{sy(pos[u][1]):.1f}" '
                         f'x2="{sx(pos[v][0]):.1f}" y2="{sy(pos[v][1]):.1f}" '
                         f'stroke="#30363d" stroke-width="1.3"/>')
        for i in range(n):
            parts.append(f'<circle cx="{sx(pos[i][0]):.1f}" cy="{sy(pos[i][1]):.1f}" r="7" '
                         f'fill="#161b22" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{pi*panel_w + panel_w/2:.0f}" y="{height-15}" fill="{col}" '
                     f'font-size="10" text-anchor="middle">{name}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
