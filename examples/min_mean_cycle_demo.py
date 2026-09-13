"""Demo: Karp's minimum mean cycle -- the tightest average-weight loop in a directed graph.

Models a small arbitrage/currency-style graph and a cyclic-schedule graph, finds the minimum mean
cycle of each in O(V*E), and draws the graph with the winning cycle highlighted.

    python examples/min_mean_cycle_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from min_mean_cycle import min_mean_cycle, cycle_mean  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Karp's minimum mean cycle: the loop of least average edge weight, in O(V*E)\n")

    # A directed graph with several cycles of different average cost.
    names = ["A", "B", "C", "D", "E"]
    edges = [
        (0, 1, 2.0), (1, 2, 2.0), (2, 0, 2.0),     # triangle A-B-C, mean 2
        (2, 3, 1.0), (3, 4, -4.0), (4, 2, 1.0),    # triangle C-D-E, total -2, mean -0.667
        (1, 3, 5.0), (3, 1, 5.0),                  # expensive 2-cycle B-D, mean 5
    ]
    n = len(names)

    print("  Directed graph, edges (from -> to : weight):")
    for u, v, w in edges:
        print(f"    {names[u]} -> {names[v]} : {w:+.1f}")

    mean, cyc = min_mean_cycle(n, edges)
    cyc_str = " -> ".join(names[c] for c in cyc) + " -> " + names[cyc[0]]
    print(f"\n  Minimum mean cycle: {cyc_str}")
    print(f"  Mean weight: {mean:.4f}  (verified: {cycle_mean(cyc, edges):.4f})")
    print(f"  {'Negative cycle detected' if mean < 0 else 'No negative cycle'} "
          f"(minimum mean {'<' if mean < 0 else '>='} 0).")

    print("\n  Karp's theorem: lambda* = min_v max_k (d_n(v) - d_k(v)) / (n - k), where d_k(v)")
    print("  is the least weight of an exactly-k-edge walk from a source. One O(V*E) DP sweep,")
    print("  no cycle enumeration. The min mean cycle is the negative-cycle certificate, the")
    print("  optimal steady-state of a cyclic schedule, and the pivot in min-cost-flow cancelling.")

    _svg(os.path.join(outdir, "min_mean_cycle.svg"), names, edges, cyc, mean)
    print(f"\n  wrote {os.path.join(outdir, 'min_mean_cycle.svg')}")


def _svg(path, names, edges, cyc, mean, width=760, height=440):
    n = len(names)
    cx, cy, R = width / 2, height / 2 + 10, 150
    pos = {}
    for i in range(n):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        pos[i] = (cx + R * math.cos(ang), cy + R * math.sin(ang))

    cyc_edges = {(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="15">'
        f'Minimum mean cycle (red), mean weight {mean:.3f}</text>',
        '<defs>'
        '<marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#8b949e"/></marker>'
        '<marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#ff6b6b"/></marker>'
        '</defs>',
    ]

    def edge_arc(x1, y1, x2, y2, hot):
        # shorten to node radius; curve slightly so bidirectional edges don't overlap
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1
        ux, uy = dx / d, dy / d
        r = 20
        sx, sy = x1 + ux * r, y1 + uy * r
        ex, ey = x2 - ux * r, y2 - uy * r
        # perpendicular offset for curvature
        px, py = -uy, ux
        mx, my = (sx + ex) / 2 + px * 22, (sy + ey) / 2 + py * 22
        col = "#ff6b6b" if hot else "#30363d"
        mk = "ar" if hot else "a"
        return (f'<path d="M{sx:.0f},{sy:.0f} Q{mx:.0f},{my:.0f} {ex:.0f},{ey:.0f}" '
                f'fill="none" stroke="{col}" stroke-width="{2.5 if hot else 1.3}" '
                f'marker-end="url(#{mk})"/>', (mx, my))

    for u, v, w in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        hot = (u, v) in cyc_edges
        seg, (lx, ly) = edge_arc(x1, y1, x2, y2, hot)
        parts.append(seg)
        col = "#ffd43b" if hot else "#8b949e"
        parts.append(f'<text x="{lx:.0f}" y="{ly:.0f}" fill="{col}" font-size="10" '
                     f'text-anchor="middle">{w:+.0f}</text>')

    for i in range(n):
        x, y = pos[i]
        on = any(i == c for c in cyc)
        col = "#ff6b6b" if on else "#4dabf7"
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="19" fill="#161b22" '
                     f'stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{x:.0f}" y="{y+5:.0f}" fill="{col}" font-size="13" '
                     f'text-anchor="middle">{names[i]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
