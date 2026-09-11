"""Demo: Floyd-Warshall -- shortest paths between every pair of nodes.

Computes the all-pairs distance matrix of a small weighted graph (checked against Dijkstra),
reconstructs a route, detects a negative cycle, and builds the transitive closure. Draws the
all-pairs distance matrix as a heatmap.

    python examples/floyd_warshall_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from floyd_warshall import (floyd_warshall, reconstruct_path,  # noqa: E402
                            has_negative_cycle, transitive_closure, dijkstra_all_pairs)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # a 6-node directed road network
    edges = [(0, 1, 7), (0, 2, 9), (0, 5, 14), (1, 2, 10), (1, 3, 15),
             (2, 3, 11), (2, 5, 2), (3, 4, 6), (5, 4, 9)]
    n = 6
    dist, nxt = floyd_warshall(n, edges)

    print("Floyd-Warshall: all-pairs shortest paths in one O(V^3) pass\n")
    print("  distance matrix (rows = from, cols = to):")
    header = "      " + "".join(f"{j:>5}" for j in range(n))
    print(header)
    for i in range(n):
        row = "".join(f"{('.' if dist[i][j] == math.inf else int(dist[i][j])):>5}" for j in range(n))
        print(f"  {i} :{row}")
    print(f"\n  shortest 0 -> 4: {reconstruct_path(nxt, 0, 4)}  (cost {int(dist[0][4])})")
    print(f"  agrees with all-pairs Dijkstra: {dist == dijkstra_all_pairs(n, edges)}\n")

    print("  Negative weights are fine (Dijkstra can't do these), and a negative loop is flagged:")
    print(f"    0->1->2->0 with weights 1,-3,1 has a negative cycle: "
          f"{has_negative_cycle(3, [(0, 1, 1), (1, 2, -3), (2, 0, 1)])}")

    tc = transitive_closure(n, [(u, v) for u, v, _ in edges])
    reachable_from_0 = [j for j in range(n) if tc[0][j]]
    print(f"\n  transitive closure: node 0 can reach {reachable_from_0}")
    print("  Three nested loops, no priority queue -- it also gives reachability (boolean OR)")
    print("  and detects negative cycles (a negative diagonal). Used for routing tables and")
    print("  network distance matrices.")

    _svg(os.path.join(outdir, "floyd_warshall.svg"), n, dist)
    print(f"\n  wrote {os.path.join(outdir, 'floyd_warshall.svg')}")


def _svg(path, n, dist, w=760, h=430):
    finite = [dist[i][j] for i in range(n) for j in range(n) if dist[i][j] != math.inf]
    dmax = max(finite) if finite else 1

    cell = 54
    x0, y0 = 110, 100
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Floyd-Warshall all-pairs distance matrix</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'cell [i][j] = shortest distance from i to j; darker = nearer, gray = unreachable</text>',
    ]

    parts.append(f'<text x="{x0 + n * cell / 2:.1f}" y="{y0 - 30:.1f}" fill="#8b949e" '
                 f'font-size="11" text-anchor="middle">to (destination)</text>')
    for j in range(n):
        parts.append(f'<text x="{x0 + j * cell + cell / 2:.1f}" y="{y0 - 8:.1f}" fill="#ffd43b" '
                     f'font-size="11" text-anchor="middle">{j}</text>')
    parts.append(f'<text x="{x0 - 42:.1f}" y="{y0 + n * cell / 2:.1f}" fill="#8b949e" '
                 f'font-size="11" text-anchor="middle" transform="rotate(-90 {x0-42:.1f} {y0 + n*cell/2:.1f})">from (source)</text>')

    for i in range(n):
        parts.append(f'<text x="{x0 - 12:.1f}" y="{y0 + i * cell + cell / 2 + 4:.1f}" fill="#4dabf7" '
                     f'font-size="11" text-anchor="end">{i}</text>')
        for j in range(n):
            x, y = x0 + j * cell, y0 + i * cell
            d = dist[i][j]
            if d == math.inf:
                fill, tcol, label = "#30363d", "#6e7681", "."
            else:
                t = d / dmax
                rr = int(0x06 + t * (0x2a - 0x06))
                gg = int(0xd6 - t * (0xd6 - 0x30))
                bb = int(0x78 + t * (0xa0 - 0x78) * 0)
                bb = int(0x50 + t * (0x22 - 0x50))
                fill = f"#{rr:02x}{max(0,gg):02x}{max(0,bb):02x}"
                tcol = "#0d1117" if t < 0.5 else "#e6edf3"
                label = str(int(d))
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{fill}" '
                         f'stroke="#0d1117" stroke-width="1"/>')
            parts.append(f'<text x="{x + cell/2 - 1:.1f}" y="{y + cell/2 + 4:.1f}" fill="{tcol}" '
                         f'font-size="12" text-anchor="middle">{label}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
