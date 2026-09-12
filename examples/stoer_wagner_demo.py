"""Demo: Stoer-Wagner global minimum cut -- the graph's weakest partition.

Finds the global min cut of the classic textbook graph and a two-cluster network, confirms against
brute force, and draws the graph with the cut edges highlighted and the two sides colored.

    python examples/stoer_wagner_demo.py [output_dir]
"""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stoer_wagner import min_cut, cut_weight  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Stoer-Wagner: the global minimum cut of a weighted graph\n")

    # the classic Stoer-Wagner example
    edges = [(0, 1, 2), (0, 4, 3), (1, 2, 3), (1, 4, 2), (1, 5, 2), (2, 3, 4),
             (2, 6, 2), (3, 6, 2), (3, 7, 2), (4, 5, 3), (5, 6, 1), (6, 7, 3)]
    n = 8
    cw, part = min_cut(n, edges)
    print(f"  Classic 8-vertex graph, {len(edges)} weighted edges:")
    print(f"    global minimum cut weight: {cw}")
    print(f"    partition: {sorted(part)} | {sorted(set(range(n)) - part)}")

    def brute(n, edges):
        best = math.inf
        for r in range(1, n):
            for combo in itertools.combinations(range(n), r):
                best = min(best, cut_weight(n, edges, set(combo)))
        return best
    print(f"    brute force confirms: {brute(n, edges)}")

    # a two-cluster network: dense clusters joined by a few thin links
    cluster = []
    for u in range(4):
        for v in range(u + 1, 4):
            cluster.append((u, v, 10))          # dense cluster A
    for u in range(4, 8):
        for v in range(u + 1, 8):
            cluster.append((u, v, 10))          # dense cluster B
    cluster += [(1, 5, 1), (2, 6, 1)]           # two thin bridges
    cw2, part2 = min_cut(8, cluster)
    print(f"\n  Two dense clusters joined by 2 weak links:")
    print(f"    min cut {cw2} -> partition {sorted(part2)} | {sorted(set(range(8)) - part2)}")
    print(f"    the cut finds exactly the weak bridges between the clusters")

    print("\n  Each phase grows a set by repeatedly adding the most tightly-connected vertex; the last")
    print("  vertex added gives a provable s-t min cut, then s and t merge. V-1 phases later, the")
    print("  smallest cut seen is the global minimum -- no max-flow, no augmenting paths.")

    _svg(os.path.join(outdir, "stoer_wagner.svg"), n, edges, part)
    print(f"\n  wrote {os.path.join(outdir, 'stoer_wagner.svg')}")


def _svg(path, n, edges, part, width=760, height=430):
    # circular layout
    cx, cy, r = width / 2, height / 2 + 20, 150
    pos = {}
    for i in range(n):
        a = -math.pi / 2 + 2 * math.pi * i / n
        pos[i] = (cx + r * math.cos(a), cy + r * math.sin(a))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Stoer-Wagner global minimum cut</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue and green vertices are the two sides; red edges cross the cut (the weakest partition)</text>',
    ]
    for u, v, wt in edges:
        crosses = (u in part) != (v in part)
        col = "#ff6b6b" if crosses else "#30363d"
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     f'stroke="{col}" stroke-width="{1.2 + wt*0.5:.1f}"/>')
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        parts.append(f'<text x="{mx:.1f}" y="{my - 3:.1f}" fill="{"#ff6b6b" if crosses else "#8b949e"}" '
                     f'font-size="10" text-anchor="middle">{wt}</text>')
    for i in range(n):
        x, y = pos[i]
        col = "#4dabf7" if i in part else "#06d6a0"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#0d1117" font-size="14" '
                     f'text-anchor="middle" font-weight="bold">{i}</text>')

    cw = cut_weight(n, edges, part)
    parts.append(f'<text x="20" y="{height-16}" fill="#ff6b6b" font-size="13">'
                 f'minimum cut weight = {cw}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
