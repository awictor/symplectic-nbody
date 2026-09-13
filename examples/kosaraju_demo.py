"""Demo: Kosaraju's two-pass SCC decomposition and the DAG condensation it produces.

Builds a directed graph with several strongly connected components, finds them with Kosaraju's
two-pass DFS, cross-checks against Tarjan, and draws the graph with vertices coloured by SCC and the
condensation drawn as a DAG.

    python examples/kosaraju_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kosaraju import strongly_connected_components, component_index, condensation  # noqa: E402
from tarjan_scc import strongly_connected_components as tarjan_scc  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kosaraju's algorithm: strongly connected components in two DFS passes\n")

    # a graph with 4 clear SCCs chained together
    n = 11
    edges = [
        (0, 1), (1, 2), (2, 0),          # SCC A = {0,1,2}
        (2, 3),                          # A -> B
        (3, 4), (4, 5), (5, 3),          # SCC B = {3,4,5}
        (5, 6),                          # B -> C
        (6, 7), (7, 6),                  # SCC C = {6,7}
        (3, 8),                          # B -> D
        (8, 9), (9, 10), (10, 8),        # SCC D = {8,9,10}
    ]

    comps = strongly_connected_components(n, edges)
    tarjan = tarjan_scc(n, edges)

    print(f"  {n} vertices, {len(edges)} directed edges")
    print(f"  Kosaraju found {len(comps)} strongly connected components:")
    for i, c in enumerate(comps):
        print(f"    SCC {i}: {c}")

    kset = frozenset(frozenset(c) for c in comps)
    tset = frozenset(frozenset(c) for c in tarjan)
    print(f"\n  matches Tarjan's decomposition: {kset == tset}")

    num, cedges = condensation(n, edges)
    print(f"\n  condensation (SCC DAG): {num} super-nodes, edges {cedges}")
    print(f"  the condensation is always acyclic -- contracting each SCC removes every cycle,")
    print(f"  which is why SCC decomposition is step one in 2-SAT and dependency analysis.")

    _svg(os.path.join(outdir, "kosaraju.svg"), n, edges, comps, num, cedges)
    print(f"\n  wrote {os.path.join(outdir, 'kosaraju.svg')}")


def _svg(path, n, edges, comps, num, cedges, width=760, height=440):
    idx = [0] * n
    for ci, c in enumerate(comps):
        for v in c:
            idx[v] = ci
    colors = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#b197fc", "#ff6b6b", "#e6edf3"]

    # place vertices grouped by SCC around circles
    import math as m
    pos = {}
    ncomp = len(comps)
    for ci, c in enumerate(comps):
        cx = 90 + (width - 320) * ci / max(1, ncomp - 1)
        cy = 150
        for j, v in enumerate(c):
            ang = 2 * m.pi * j / len(c)
            pos[v] = (cx + 26 * m.cos(ang), cy + 26 * m.sin(ang))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Kosaraju SCCs (top, coloured by component) and the DAG condensation (bottom)</text>',
        '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#8b949e"/></marker></defs>',
    ]
    # edges
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # shorten toward target for arrowhead
        dx, dy = x2 - x1, y2 - y1
        d = m.hypot(dx, dy) or 1
        x2s, y2s = x2 - 8 * dx / d, y2 - 8 * dy / d
        same = idx[u] == idx[v]
        col = colors[idx[u] % len(colors)] if same else "#484f58"
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}" '
                     f'stroke="{col}" stroke-width="1.5" marker-end="url(#ar)"/>')
    for v in range(n):
        x, y = pos[v]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{colors[idx[v] % len(colors)]}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#0d1117" font-size="10" '
                     f'text-anchor="middle">{v}</text>')

    # condensation DAG at the bottom
    cyc = 340
    cpos = {}
    for ci in range(num):
        cx = 90 + (width - 320) * ci / max(1, num - 1)
        cpos[ci] = (cx, cyc)
    parts.append(f'<text x="20" y="{cyc-40}" fill="#8b949e" font-size="11">condensation:</text>')
    for a, b in cedges:
        x1, y1 = cpos[a]
        x2, y2 = cpos[b]
        dx, dy = x2 - x1, y2 - y1
        d = m.hypot(dx, dy) or 1
        x2s, y2s = x2 - 14 * dx / d, y2 - 14 * dy / d
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}" '
                     f'stroke="#8b949e" stroke-width="1.5" marker-end="url(#ar)"/>')
    for ci in range(num):
        x, y = cpos[ci]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="{colors[ci % len(colors)]}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#0d1117" font-size="10" '
                     f'text-anchor="middle">S{ci}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
