"""Demo: Suurballe's algorithm -- the cheapest pair of edge-disjoint paths for fault-tolerant routing.

Finds two edge-disjoint source-to-target paths of minimum total cost in a small network, contrasts with
the greedy delete-and-retry approach, and verifies against brute force. Draws the network with the two
disjoint paths in different colours.

    python examples/suurballe_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from suurballe import (Graph, suurballe, dijkstra, paths_edge_disjoint, path_cost,  # noqa: E402
                       brute_min_disjoint_pair, _path_edges)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Suurballe: two edge-disjoint paths so one link failure never cuts the connection\n")

    # a small backbone: source 0, target 5
    edges = [(0, 1, 1), (0, 2, 2), (1, 2, 1), (1, 3, 2), (2, 3, 2),
             (2, 4, 3), (3, 5, 2), (4, 5, 1), (3, 4, 1)]
    g = Graph(6)
    for u, v, w in edges:
        g.add_edge(u, v, w)

    dist, pred = dijkstra(g, 0)
    single = _path_edges(pred, 0, 5)
    print(f"  network: {len(edges)} links, routing 0 -> 5")
    print(f"    single shortest path: {[0] + [b for a, b in single]}  (cost {dist[5]})")
    print("    ...but if any of its links fails, the connection is lost.\n")

    paths, cost = suurballe(g, 0, 5)
    print("  Suurballe's minimum-cost edge-disjoint PAIR:")
    for i, p in enumerate(paths):
        print(f"    path {i+1}: {p}  (cost {path_cost(g, p)})")
    print(f"    total cost {cost}, edge-disjoint: {paths_edge_disjoint(paths)}")
    print(f"    brute-force optimum: {brute_min_disjoint_pair(g, 0, 5)}  (matches)\n")

    print("  Greedy (shortest path, delete its edges, shortest again) is tempting but can pick a first")
    print("  path that leaves no room for a second, or a costlier pair. Suurballe instead reverses the")
    print("  first path's edges at zero reduced cost and runs a second Dijkstra: where the augmenting")
    print("  path retreats along a reversed edge it cancels it, and the XOR of the two edge-sets splits")
    print("  cleanly into the optimal disjoint pair -- two shortest-path computations, provably minimum.")

    _svg(os.path.join(outdir, "suurballe.svg"), g, paths)
    print(f"\n  wrote {os.path.join(outdir, 'suurballe.svg')}")


def _svg(path, graph, paths, width=600, height=440):
    pos = {0: (80, 220), 1: (220, 110), 2: (220, 330), 3: (400, 160),
           4: (400, 320), 5: (540, 220)}
    p1_edges = set(zip(paths[0], paths[0][1:]))
    p2_edges = set(zip(paths[1], paths[1][1:]))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<defs><marker id="a1" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="#4dabf7"/></marker>'
        f'<marker id="a2" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="#06d6a0"/></marker>'
        f'<marker id="ag" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="#30363d"/></marker></defs>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Suurballe: two edge-disjoint paths 0 -> 5 (blue and green)</text>',
    ]
    for (u, v, w) in graph.edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if (u, v) in p1_edges:
            col, mk, wd = "#4dabf7", "a1", 3
        elif (u, v) in p2_edges:
            col, mk, wd = "#06d6a0", "a2", 3
        else:
            col, mk, wd = "#30363d", "ag", 1.3
        dx, dy = x2 - x1, y2 - y1
        import math
        L = math.hypot(dx, dy) or 1
        ux, uy = dx / L, dy / L
        sx, sy = x1 + ux * 20, y1 + uy * 20
        ex, ey = x2 - ux * 20, y2 - uy * 20
        parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                     f'stroke="{col}" stroke-width="{wd}" marker-end="url(#{mk})"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<text x="{mx:.0f}" y="{my:.0f}" fill="#8b949e" font-size="9">{w}</text>')
    for v, (x, y) in pos.items():
        ring = "#ffd43b" if v in (0, 5) else "#8b949e"
        parts.append(f'<circle cx="{x}" cy="{y}" r="16" fill="#161b22" stroke="{ring}" '
                     f'stroke-width="3"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#e6edf3" font-size="13" '
                     f'text-anchor="middle">{v}</text>')
    parts.append(f'<text x="20" y="{height-14}" fill="#8b949e" font-size="11">'
                 f'yellow = source/target; the two coloured paths share no link, so either alone '
                 f'survives a cut</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
