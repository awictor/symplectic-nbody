"""Demo: minimum spanning arborescence -- the cheapest one-way broadcast tree from a root.

Finds the minimum-cost directed spanning tree rooted at a source, shows why the greedy "cheapest
incoming edge" can fail (it makes a cycle), and draws the graph with the chosen arborescence edges
highlighted.

    python examples/arborescence_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arborescence import min_arborescence, brute_min_arborescence  # noqa: E402

# a directed network: root 0 broadcasting to 1..5, with a tempting cheap cycle among 1,2,3
N = 6
EDGES = [
    (0, 1, 10), (0, 2, 12),
    (1, 2, 2), (2, 3, 3), (3, 1, 4),      # cheap cycle 1->2->3->1
    (2, 4, 6), (3, 5, 5),
    (4, 5, 8), (1, 4, 9),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    root = 0
    total, parent = min_arborescence(N, EDGES, root)

    print("Minimum spanning arborescence: cheapest one-way broadcast tree\n")
    print(f"  {N} nodes, {len(EDGES)} directed edges, root = {root}\n")

    print("  greedy 'cheapest incoming edge per node' would pick:")
    best_in = {}
    for u, v, w in EDGES:
        if v != root and (v not in best_in or w < best_in[v][2]):
            best_in[v] = (u, v, w)
    for v in sorted(best_in):
        print(f"    node {v} <- node {best_in[v][0]} (cost {best_in[v][2]})")
    print("    ...but 1<-3, 2<-1, 3<-2 form a CYCLE (1->2->3->1), not a tree -- greedy fails.\n")

    print(f"  Chu-Liu/Edmonds minimum arborescence (total cost {total}):")
    for v in range(N):
        if parent[v] is not None:
            u, _, w = parent[v]
            print(f"    node {v} <- node {u} (cost {w})")

    bw, _ = brute_min_arborescence(N, EDGES, root)
    print(f"\n  verified against brute force over all arborescences: {total == bw}")
    print("\n  The algorithm contracts the cheap cycle into a super-node, reweights edges entering it")
    print("  by what they'd save, and recurses -- then expands, breaking each cycle at the one node")
    print("  reached from outside. This is the directed analogue of the minimum spanning tree, where")
    print("  Kruskal and Prim don't apply because edge directions constrain the tree.")

    _svg(os.path.join(outdir, "arborescence.svg"), parent)
    print(f"\n  wrote {os.path.join(outdir, 'arborescence.svg')}")


def _svg(path, parent, width=760, height=440):
    # circular-ish fixed layout
    pos = {
        0: (110, 220),
        1: (300, 90), 2: (300, 350),
        3: (480, 220),
        4: (620, 110), 5: (640, 340),
    }
    chosen = set()
    for v in range(N):
        if parent[v] is not None:
            chosen.add((parent[v][0], v))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<defs>'
        '<marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L8,3 L0,6 Z" fill="#8b949e"/></marker>'
        '<marker id="arG" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L8,3 L0,6 Z" fill="#06d6a0"/></marker>'
        '</defs>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Minimum spanning arborescence from node 0 (green = chosen edges)</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'every node gets exactly one incoming green edge; the total cost is minimum</text>',
    ]

    def edge_line(u, v, w, col, marker, width_px):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1
        # trim to node radius
        r = 20
        sx, sy = x1 + dx / d * r, y1 + dy / d * r
        ex, ey = x2 - dx / d * r, y2 - dy / d * r
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                     f'stroke="{col}" stroke-width="{width_px}" marker-end="url(#{marker})"/>')
        parts.append(f'<text x="{mx:.0f}" y="{my-4:.0f}" fill="{col}" font-size="11" '
                     f'text-anchor="middle">{w}</text>')

    for u, v, w in EDGES:
        if (u, v) in chosen:
            edge_line(u, v, w, "#06d6a0", "arG", 3)
        else:
            edge_line(u, v, w, "#8b949e", "ar", 1)

    for v, (x, y) in pos.items():
        col = "#ffd43b" if v == 0 else "#4dabf7"
        parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{col}"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#0d1117" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')
    parts.append(f'<text x="{pos[0][0]:.0f}" y="{pos[0][1]-28:.0f}" fill="#ffd43b" '
                 f'font-size="11" text-anchor="middle">root</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
