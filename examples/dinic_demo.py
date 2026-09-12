"""Demo: Dinic's algorithm -- maximum flow through a network via layered blocking flows.

Computes the max flow of a small pipe network, shows the BFS level graph that Dinic builds each phase,
recovers the min cut, and draws the network with the saturated cut edges highlighted.

    python examples/dinic_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dinic import Dinic, edmonds_karp  # noqa: E402

# a pipe network from source S(0) to sink T(5)
EDGES = [
    (0, 1, 10), (0, 2, 10),
    (1, 2, 2), (1, 3, 4), (1, 4, 8),
    (2, 4, 9),
    (3, 5, 10), (4, 3, 6), (4, 5, 10),
]
N = 6
NAMES = ["S", "a", "b", "c", "d", "T"]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    d = Dinic(N)
    # remember forward-edge slots to read the flow used on each
    slots = []
    for u, v, c in EDGES:
        slots.append((u, len(d.graph[u]), c))
        d.add_edge(u, v, c)

    flow = d.max_flow(0, 5)
    side = d.min_cut(0)

    print("Dinic's algorithm: maximum flow by layered blocking flows\n")
    print(f"  {N} nodes, {len(EDGES)} directed pipes, source S sink T\n")
    print(f"  maximum flow: {flow} units  (Edmonds-Karp agrees: "
          f"{flow == edmonds_karp(N, EDGES, 0, 5)})\n")

    print("  flow on each pipe (used / capacity):")
    edge_used = {}
    for (u, v, c), (su, si, cap) in zip(EDGES, slots):
        used = cap - d.graph[su][si][1]
        edge_used[(u, v)] = used
        print(f"    {NAMES[u]} -> {NAMES[v]}:  {used}/{c}")

    print(f"\n  minimum cut: source side {{{', '.join(NAMES[v] for v in side)}}}")
    cut_edges = [(u, v, c) for (u, v, c) in EDGES if u in set(side) and v not in set(side)]
    cut_cap = sum(c for _, _, c in cut_edges)
    print(f"  cut edges: {[(NAMES[u], NAMES[v], c) for u, v, c in cut_edges]}")
    print(f"  cut capacity {cut_cap} == max flow {flow}  (max-flow min-cut theorem)")

    print("\n  Each phase: a BFS assigns every node its distance from S (the level graph), then a DFS")
    print("  pushes flow only along level-increasing edges until blocked, saturating an edge on each")
    print("  path. The sink's level strictly rises per phase, so only O(V) phases run -- far fewer")
    print("  augmentations than one-path-at-a-time Ford-Fulkerson.")

    _svg(os.path.join(outdir, "dinic.svg"), edge_used, set(side))
    print(f"\n  wrote {os.path.join(outdir, 'dinic.svg')}")


def _svg(path, edge_used, side, width=760, height=400):
    pos = {0: (60, 200), 1: (250, 90), 2: (250, 310),
           3: (500, 90), 4: (500, 310), 5: (700, 200)}
    caps = {(u, v): c for u, v, c in EDGES}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<defs>'
        '<marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L8,3 L0,6 Z" fill="#8b949e"/></marker>'
        '</defs>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f"Dinic max flow: pipe thickness = flow used, red pipes cross the min cut</text>",
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'each label is used/capacity; blue nodes are the source side of the minimum cut</text>',
    ]

    import math
    for (u, v), c in caps.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        dd = math.hypot(dx, dy) or 1
        r = 22
        sx, sy = x1 + dx / dd * r, y1 + dy / dd * r
        ex, ey = x2 - dx / dd * r, y2 - dy / dd * r
        used = edge_used.get((u, v), 0)
        is_cut = u in side and v not in side
        if is_cut:
            col = "#ff6b6b"
        elif used > 0:
            col = "#06d6a0"
        else:
            col = "#484f58"
        wid = 1.5 + 1.8 * used ** 0.5
        parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                     f'stroke="{col}" stroke-width="{wid:.1f}" marker-end="url(#a)"/>')
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        parts.append(f'<text x="{mx:.0f}" y="{my-4:.0f}" fill="#ffd43b" font-size="10" '
                     f'text-anchor="middle">{used}/{c}</text>')

    for v, (x, y) in pos.items():
        fill = "#4dabf7" if v in side else "#8b949e"
        parts.append(f'<circle cx="{x}" cy="{y}" r="22" fill="{fill}"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#0d1117" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{NAMES[v]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
