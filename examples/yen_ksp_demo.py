"""Demo: Yen's K shortest paths -- the best route and its next-best alternatives.

Finds the K cheapest loopless routes between two nodes of a small road network, lists them in order,
and draws the graph with the top three routes highlighted in different colours.

    python examples/yen_ksp_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yen_ksp import k_shortest_paths, brute_k_shortest  # noqa: E402

# a small directed road network with several viable routes from A(0) to F(5)
NAMES = ["A", "B", "C", "D", "E", "F"]
EDGES = [
    (0, 1, 4), (0, 2, 2),
    (1, 2, 1), (1, 3, 5),
    (2, 1, 1), (2, 3, 8), (2, 4, 10),
    (3, 4, 2), (3, 5, 6),
    (4, 5, 3),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    K = 4
    paths = k_shortest_paths(6, EDGES, 0, 5, K)

    print("Yen's algorithm: the K shortest loopless routes from A to F\n")
    print(f"  network: {len(EDGES)} one-way roads among {len(NAMES)} junctions\n")

    print(f"  {K} shortest routes (increasing cost):")
    for rank, (cost, path) in enumerate(paths, 1):
        route = " -> ".join(NAMES[v] for v in path)
        print(f"    {rank}. cost {cost:2d}:  {route}")

    brute = brute_k_shortest(6, EDGES, 0, 5, K)
    print(f"\n  matches brute-force enumeration of all simple paths: "
          f"{[c for c, _ in paths] == [c for c, _ in brute]}")

    print("\n  Each next route is found by taking a prefix of the previous one, banning the edges")
    print("  already used out of the spur node (to force a different continuation) and the earlier")
    print("  nodes (to stay loopless), then running Dijkstra from the spur to the target. The cheapest")
    print("  such candidate becomes the next route -- alternatives that are genuinely distinct, not")
    print("  the best path padded with detours.")

    _svg(os.path.join(outdir, "yen_ksp.svg"), paths)
    print(f"\n  wrote {os.path.join(outdir, 'yen_ksp.svg')}")


def _svg(path, paths, width=760, height=430):
    pos = {0: (80, 220), 1: (280, 100), 2: (280, 340),
           3: (500, 100), 4: (500, 340), 5: (690, 220)}
    colors = ["#06d6a0", "#ffd43b", "#ff922b", "#b197fc"]

    # map each edge to the best (lowest-rank) route that uses it, for colouring
    edge_rank = {}
    for rank, (_, p) in enumerate(paths):
        for u, v in zip(p, p[1:]):
            edge_rank.setdefault((u, v), rank)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<defs>'
        '<marker id="g" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L8,3 L0,6 Z" fill="#8b949e"/></marker>'
        '</defs>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f"Yen's K shortest routes A -> F (route 1 green, then yellow, orange, purple)</text>",
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'grey roads are unused by the top routes; a road is coloured by the best route that takes it'
        f'</text>',
    ]

    import math
    for u, v, w in EDGES:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1
        r = 22
        sx, sy = x1 + dx / d * r, y1 + dy / d * r
        ex, ey = x2 - dx / d * r, y2 - dy / d * r
        rank = edge_rank.get((u, v))
        if rank is not None and rank < len(colors):
            col, wid = colors[rank], 3.5 - rank * 0.5
            parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                         f'stroke="{col}" stroke-width="{wid:.1f}"/>')
        else:
            parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                         f'stroke="#484f58" stroke-width="1" marker-end="url(#g)"/>')
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        parts.append(f'<text x="{mx:.0f}" y="{my-4:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{w}</text>')

    for v, (x, y) in pos.items():
        fill = "#4dabf7"
        if v == 0 or v == 5:
            fill = "#ff6b6b"
        parts.append(f'<circle cx="{x}" cy="{y}" r="22" fill="{fill}"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#0d1117" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{NAMES[v]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
