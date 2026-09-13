"""Demo: Edmonds' blossom algorithm -- maximum matching in a general graph with odd cycles.

Matches vertices in graphs a bipartite matcher cannot handle -- odd cycles and the Petersen graph --
and shows the matching is maximum and valid. Draws a graph with its matched edges highlighted.

    python examples/blossom_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blossom import (Graph, maximum_matching, matching_size, matched_pairs,  # noqa: E402
                     is_valid_matching, brute_maximum_matching_size)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Edmonds' blossom: maximum matching in a general graph (odd cycles and all)\n")

    print("  odd cycles -- where naive bipartite matching fails:")
    for k in (3, 5, 7):
        g = Graph(k)
        for i in range(k):
            g.add_edge(i, (i + 1) % k)
        m = maximum_matching(g)
        print(f"    {k}-cycle: matched {matching_size(m)} edges "
              f"(max possible floor({k}/2) = {k//2}), valid = {is_valid_matching(g, m)}")
    print()

    # Petersen graph
    pet = Graph(10)
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    for u, v in outer + spokes + inner:
        pet.add_edge(u, v)
    m = maximum_matching(pet)
    print(f"  Petersen graph (10 vertices, 3-regular): perfect matching of {matching_size(m)} edges")
    print(f"    matched pairs: {matched_pairs(m)}\n")

    # a graph with a blossom: a triangle hanging off a path
    g = Graph(6)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2), (0, 5)]:  # 2-3-4 triangle
        g.add_edge(u, v)
    m = maximum_matching(g)
    print(f"  triangle-with-tails graph: matched {matching_size(m)} edges "
          f"(brute force: {brute_maximum_matching_size(g)})\n")

    print("  A blossom is an odd cycle reached by an alternating path. Edmonds contracts each one into")
    print("  a super-vertex, finds an augmenting path in the smaller graph, then expands the blossom to")
    print("  thread the path back -- turning general matching into the bipartite augmenting-path hunt,")
    print("  and founding the theory of polynomial-time algorithms along the way.")

    _svg(os.path.join(outdir, "blossom.svg"), pet, m)
    print(f"\n  wrote {os.path.join(outdir, 'blossom.svg')}")


def _svg(path, graph, match, width=520, height=520):
    n = graph.n
    cx, cy = width / 2, height / 2 + 10
    # Petersen layout: outer pentagon (0-4), inner pentagram (5-9)
    pos = {}
    for i in range(5):
        ang = -math.pi / 2 + i * 2 * math.pi / 5
        pos[i] = (cx + 180 * math.cos(ang), cy + 180 * math.sin(ang))
        pos[5 + i] = (cx + 90 * math.cos(ang), cy + 90 * math.sin(ang))

    matched = set(matched_pairs(match))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="16">'
        f'Petersen graph: a perfect matching (green) found by blossom</text>',
    ]
    # edges
    for u, v in graph._edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if (u, v) in matched:
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#06d6a0" stroke-width="4"/>')
        else:
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#30363d" stroke-width="1.5"/>')
    # vertices
    for v in range(n):
        x, y = pos[v]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="#161b22" stroke="#4dabf7" '
                     f'stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#e6edf3" font-size="12" '
                     f'text-anchor="middle">{v}</text>')
    parts.append(f'<text x="20" y="{height-14}" fill="#8b949e" font-size="11">'
                 f'every vertex is covered exactly once -- a perfect matching of a graph full of '
                 f'odd cycles</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
