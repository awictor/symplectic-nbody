"""Demo: maximum flow, the min-cut theorem, and bipartite matching.

Computes the max flow of a classic network, extracts the minimum cut (confirming max-flow ==
min-cut), and solves a bipartite matching by reduction to flow. Draws the network with edge
capacities and the min-cut edges highlighted.

    python examples/max_flow_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from max_flow import edmonds_karp, min_cut_value, bipartite_matching  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Maximum flow: Edmonds-Karp, min-cut, and bipartite matching\n")

    # the classic CLRS network
    edges = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12), (3, 2, 9),
             (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    flow = edmonds_karp(6, edges, 0, 5)
    fval, cut = min_cut_value(6, edges, 0, 5)
    print(f"  A 6-node network (source 0, sink 5):")
    print(f"    maximum flow: {flow}")
    print(f"    minimum cut edges: {[(u, v) for u, v, _ in cut]}")
    print(f"    min-cut capacity: {sum(c for _, _, c in cut)}  == max flow: "
          f"{sum(c for _, _, c in cut) == flow}")
    print("    (the max-flow min-cut theorem: the bottleneck edges limit the whole network)\n")

    # a few small networks
    print("  Small networks:")
    print(f"    two parallel pipes (5, 3):        {edmonds_karp(2, [(0, 1, 5), (0, 1, 3)], 0, 1)}")
    print(f"    series with a bottleneck (10, 3): {edmonds_karp(3, [(0, 1, 10), (1, 2, 3)], 0, 2)}")
    tricky = [(0, 1, 1), (0, 2, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)]
    print(f"    anti-greedy (needs rerouting):    {edmonds_karp(4, tricky, 0, 3)} "
          f"(residual reverse edges reroute flow)\n")

    # bipartite matching as flow
    print("  Bipartite matching by reduction to max flow (jobs to workers):")
    workers = ["Ann", "Bob", "Cy", "Dot"]
    jobs = ["cook", "clean", "drive", "shop"]
    prefs = [("Ann", "cook"), ("Ann", "clean"), ("Bob", "clean"), ("Bob", "drive"),
             ("Cy", "drive"), ("Cy", "shop"), ("Dot", "shop"), ("Dot", "cook")]
    size, pairs = bipartite_matching(workers, jobs, prefs)
    print(f"    {len(prefs)} allowed assignments -> maximum matching of {size}:")
    for w, j in sorted(pairs):
        print(f"      {w:>4} -> {j}")

    print("\n  Ford-Fulkerson pushes flow along augmenting paths and updates a residual graph where")
    print("  each used edge gains a reverse edge (so later paths can cancel flow); Edmonds-Karp")
    print("  always takes the shortest such path (BFS) for O(V E^2) time. When no augmenting path")
    print("  remains, the source-reachable residual set defines the minimum cut = the maximum flow.")

    _svg(os.path.join(outdir, "max_flow.svg"), edges, cut, flow)
    print(f"\n  wrote {os.path.join(outdir, 'max_flow.svg')}")


def _svg(path, edges, cut, flow, width=760, height=400):
    # a hand-placed layout for the 6-node network
    pos = {0: (60, 200), 1: (280, 90), 2: (280, 310), 3: (520, 90), 4: (520, 310), 5: (720, 200)}
    cutset = {(u, v) for u, v, _ in cut}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Max flow = {flow}: network with capacities; min-cut edges in red</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'source = node 0 (left), sink = node 5 (right); red edges are the bottleneck cut</text>',
        '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#484f58"/></marker>'
        '<marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#ff6b6b"/></marker></defs>',
    ]
    for u, v, c in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        import math
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        ax, ay = x0 + dx / L * 22, y0 + dy / L * 22
        bx, by = x1 - dx / L * 26, y1 - dy / L * 26
        iscut = (u, v) in cutset
        col = "#ff6b6b" if iscut else "#30363d"
        mk = "url(#ar)" if iscut else "url(#a)"
        parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                     f'stroke="{col}" stroke-width="{2.4 if iscut else 1.4}" marker-end="{mk}"/>')
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        parts.append(f'<text x="{mx:.1f}" y="{my - 4:.1f}" fill="{"#ff6b6b" if iscut else "#8b949e"}" '
                     f'font-size="11" text-anchor="middle">{c}</text>')
    for node, (x, y) in pos.items():
        fill = "#06d6a0" if node == 0 else ("#ffd43b" if node == 5 else "#4dabf7")
        parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{fill}" stroke="#0d1117" stroke-width="2"/>')
        parts.append(f'<text x="{x}" y="{y + 5}" fill="#0d1117" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{node}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
