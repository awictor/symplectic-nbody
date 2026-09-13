"""Demo: the Chinese Postman problem -- shortest closed walk covering every edge.

Solves route inspection on a small street network: identifies the odd-degree junctions, pairs them by
shortest paths to minimise retracing, and reports the optimal postman route length. Draws the graph
with odd vertices and duplicated (retraced) paths highlighted.

    python examples/chinese_postman_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chinese_postman import (chinese_postman, eulerian_circuit,  # noqa: E402
                             _build_adj, _all_pairs_shortest)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Chinese Postman: the shortest closed walk down every street\n")

    # a small street network (junctions 0..5)
    edges = [
        (0, 1, 4), (0, 2, 3), (1, 2, 2), (1, 3, 5),
        (2, 3, 3), (3, 4, 6), (2, 4, 4), (4, 5, 2), (3, 5, 5),
    ]
    n = 6
    r = chinese_postman(n, edges)
    total = r["total_edge_weight"]
    print(f"  network: {len(edges)} streets, total length {total}")
    print(f"    odd-degree junctions (dead-ends for parity): {r['odd_vertices']}")
    if r["is_eulerian"]:
        print("    already Eulerian -- the postman walks each street once, no retracing")
    else:
        print(f"    must retrace {r['extra_cost']} of length to fix parity "
              f"(min-cost pairing of odd junctions)")
    print(f"    optimal postman route length: {r['route_length']}  "
          f"= {total} (all streets) + {r['extra_cost']} (retraced)\n")

    # a fully-even variant: an actual circuit
    even_edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1), (0, 2, 2), (1, 3, 2)]
    # make degrees even by duplicating the diagonals
    even_edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]
    c = eulerian_circuit(4, even_edges)
    print(f"  an Eulerian block (square): the circuit is {c}")
    print(f"    (every street once, back to the start, zero retracing)\n")

    print("  Euler: a connected graph has a walk using every edge exactly once iff every junction has")
    print("  even degree. Odd junctions force retracing; the cheapest fix pairs them up along shortest")
    print("  paths -- a minimum-weight matching -- so the postman's optimal route is simply the sum of")
    print("  all streets plus that minimum matching cost. Polynomial-time, unlike its cousin the TSP.")

    _svg(os.path.join(outdir, "chinese_postman.svg"), n, edges, r)
    print(f"\n  wrote {os.path.join(outdir, 'chinese_postman.svg')}")


def _svg(path, n, edges, result, width=600, height=440):
    # fixed layout for 6 junctions
    pos = {0: (120, 120), 1: (300, 70), 2: (250, 230), 3: (430, 180),
           4: (400, 340), 5: (540, 300)}
    odds = set(result["odd_vertices"])

    # find the matched pairs (recompute the pairing that achieves the min for display)
    adj, degree, _, present = _build_adj(n, edges)
    dist = _all_pairs_shortest(n, adj)
    odd_list = sorted(odds)
    best = {"cost": float("inf"), "pairs": []}

    def rec(rem, acc, pairs):
        if not rem:
            if acc < best["cost"]:
                best["cost"] = acc
                best["pairs"] = pairs[:]
            return
        first = rem[0]
        for i in range(1, len(rem)):
            rec(rem[1:i] + rem[i + 1:], acc + dist[first][rem[i]], pairs + [(first, rem[i])])
    rec(odd_list, 0.0, [])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Chinese Postman: odd junctions (red) paired to minimise retracing</text>',
    ]
    # edges
    for u, v, w in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#30363d" '
                     f'stroke-width="2"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<text x="{mx:.0f}" y="{my:.0f}" fill="#8b949e" font-size="10">{w}</text>')
    # matched pairs (retraced shortest paths) as dashed arcs
    for a, b in best["pairs"]:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ffd43b" '
                     f'stroke-width="3" stroke-dasharray="6 4" opacity="0.8"/>')
    # nodes
    for v in range(n):
        x, y = pos[v]
        col = "#ff6b6b" if v in odds else "#4dabf7"
        parts.append(f'<circle cx="{x}" cy="{y}" r="16" fill="#161b22" stroke="{col}" '
                     f'stroke-width="3"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#e6edf3" font-size="13" '
                     f'text-anchor="middle">{v}</text>')
    parts.append(f'<text x="20" y="{height-16}" fill="#8b949e" font-size="11">'
                 f'route = {result["total_edge_weight"]:.0f} (all streets) + '
                 f'{result["extra_cost"]:.0f} (yellow retraced) = {result["route_length"]:.0f}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
