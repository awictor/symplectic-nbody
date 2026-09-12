"""Demo: Tarjan's strongly connected components and the condensation DAG.

Finds the SCCs of a directed graph, collapses them into the condensation (always a DAG), and shows
the topological order. Draws the graph with each SCC in its own color and the condensation beside it.

    python examples/tarjan_scc_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tarjan_scc import strongly_connected_components, condensation, topological_sort  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Tarjan's SCC: collapsing a directed graph's cycles into a DAG\n")

    # a graph with three cycles feeding forward
    edges = [(0, 1), (1, 2), (2, 0),          # SCC A: {0,1,2}
             (2, 3), (3, 4), (4, 3),           # SCC B: {3,4}
             (4, 5), (5, 6), (6, 7), (7, 5),   # SCC C: {5,6,7}
             (3, 6)]                            # cross edge
    n = 8

    comps = strongly_connected_components(n, edges)
    print(f"  {n} vertices, {len(edges)} edges")
    print(f"  strongly connected components (reverse topological order):")
    for i, c in enumerate(comps):
        print(f"    component {i}: {sorted(c)}")

    comp_of, dag, _ = condensation(n, edges)
    print(f"\n  condensation DAG super-edges: {sorted(dag)}")
    topo = topological_sort(len(comps), list(dag))
    print(f"  condensation topological order: {topo}")
    print(f"  condensation is acyclic: {topological_sort(len(comps), list(dag)) is not None}")

    # cycle detection on the original vs the condensation
    from tarjan_scc import has_cycle
    print(f"\n  original graph has a cycle: {has_cycle(n, edges)}")
    print(f"  condensation has a cycle:   {has_cycle(len(comps), list(dag))}")

    print("\n  A single depth-first search assigns each vertex a discovery index and a low-link (the")
    print("  smallest index reachable via one back-edge from its subtree). A vertex whose low-link")
    print("  equals its index is an SCC root; the stack above it forms the component. O(V+E).")

    _svg(os.path.join(outdir, "tarjan_scc.svg"), n, edges, comp_of, comps, dag)
    print(f"\n  wrote {os.path.join(outdir, 'tarjan_scc.svg')}")


def _svg(path, n, edges, comp_of, comps, dag, width=760, height=440):
    colors = ["#4dabf7", "#ffd43b", "#ff6b6b", "#06d6a0", "#b197fc", "#ff922b"]

    # left: original graph laid out on a circle, nodes colored by SCC
    cx, cy, r = 200, 250, 150
    pos = {}
    for v in range(n):
        a = -math.pi / 2 + 2 * math.pi * v / n
        pos[v] = (cx + r * math.cos(a), cy + r * math.sin(a))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Tarjan SCC: cycles colored (left), condensation DAG (right)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each colour is one strongly connected component; the condensation is always acyclic</text>',
        '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" '
        'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#484f58"/></marker></defs>',
    ]

    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1
        ax, ay = x0 + dx / L * 18, y0 + dy / L * 18
        bx, by = x1 - dx / L * 22, y1 - dy / L * 22
        # curve slightly
        mx, my = (x0 + x1) / 2 - dy / L * 14, (y0 + y1) / 2 + dx / L * 14
        parts.append(f'<path d="M{ax:.1f},{ay:.1f} Q{mx:.1f},{my:.1f} {bx:.1f},{by:.1f}" '
                     f'fill="none" stroke="#484f58" stroke-width="1.3" marker-end="url(#a)"/>')
    for v in range(n):
        x, y = pos[v]
        col = colors[comp_of[v] % len(colors)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#0d1117" font-size="14" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    # right: condensation DAG (components as super-nodes, laid out by topological order)
    topo = topological_sort(len(comps), list(dag)) or list(range(len(comps)))
    cpos = {}
    rx = 560
    for rank, cid in enumerate(topo):
        cpos[cid] = (rx, 110 + rank * 90)
    for (a, b) in dag:
        x0, y0 = cpos[a]
        x1, y1 = cpos[b]
        parts.append(f'<line x1="{x0:.1f}" y1="{y0+22:.1f}" x2="{x1:.1f}" y2="{y1-22:.1f}" '
                     f'stroke="#8b949e" stroke-width="1.6" marker-end="url(#a)"/>')
    for cid, (x, y) in cpos.items():
        col = colors[cid % len(colors)]
        members = sorted(comps[cid])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#0d1117" font-size="11" '
                     f'text-anchor="middle" font-weight="bold">{",".join(map(str, members))}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
