"""Demo: bridges and articulation points -- the single points of failure in a network.

Builds a small network of clusters joined by thin links, finds every bridge (edge whose loss
disconnects the graph) and articulation point (vertex whose loss does), and draws the network with
the failure points highlighted in red.

    python examples/bridges_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bridges import (Graph, find_bridges_and_articulation,  # noqa: E402
                     two_edge_connected_components)


# a network: three well-connected clusters joined by thin single links + one dangling node
EDGES = [
    (0, 1), (1, 2), (2, 0),          # cluster A (triangle)
    (3, 4), (4, 5), (5, 3), (3, 5),  # cluster B (dense)
    (6, 7), (7, 8), (8, 6),          # cluster C (triangle)
    (2, 3),                          # bridge A--B
    (5, 6),                          # bridge B--C
    (8, 9),                          # dangling leaf 9 off cluster C
]
N = 10


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    g = Graph(N)
    for u, v in EDGES:
        g.add_edge(u, v)

    bridges, arts = find_bridges_and_articulation(g)
    tecc = two_edge_connected_components(g)

    print("Bridges and articulation points: a 10-node, 3-cluster network\n")
    print(f"  {N} nodes, {len(EDGES)} edges, 3 clusters joined by thin links\n")

    print(f"  Bridges (edge whose loss splits the network): {len(bridges)}")
    for u, v in bridges:
        print(f"    {u} -- {v}")
    print(f"\n  Articulation points (node whose loss splits the network): {arts}")
    print("\n  2-edge-connected components (groups that survive any single link failure):")
    for i, c in enumerate(tecc):
        print(f"    group {i}: {c}")

    print("\n  Reliability reading: every bridge and cut vertex is a single point of failure. The")
    print("  dense cluster B (with its extra 3--5 chord) has no internal bridge, but the two links")
    print("  wiring the clusters together (2--3, 5--6) and the leaf link (8--9) are all bridges --")
    print("  cut any one and the network splits. Tarjan finds them in one O(V+E) DFS pass; the naive")
    print("  check would delete each edge and recount components, O(V*(V+E)).")

    _svg(os.path.join(outdir, "bridges.svg"), g, bridges, arts)
    print(f"\n  wrote {os.path.join(outdir, 'bridges.svg')}")


def _svg(path, g, bridges, arts, width=760, height=460):
    # fixed layout: three clusters left / center-top / right, leaf far right
    pos = {
        0: (120, 130), 1: (60, 240), 2: (180, 240),        # cluster A
        3: (330, 120), 4: (300, 250), 5: (410, 210),        # cluster B
        6: (560, 130), 7: (520, 250), 8: (620, 250),        # cluster C
        9: (700, 350),                                      # leaf
    }
    bridge_set = set(bridges)
    art_set = set(arts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Network failure points: red edges are bridges, red nodes are cut vertices</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'cut any red edge (or remove any red node) and the network splits into disconnected pieces'
        f'</text>',
    ]

    # edges first (under nodes)
    for u, v in g.edges:
        a, b = (u, v) if u < v else (v, u)
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if (a, b) in bridge_set:
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                         f'stroke="#ff6b6b" stroke-width="3.5"/>')
        else:
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                         f'stroke="#4dabf7" stroke-width="1.6" opacity="0.8"/>')

    # nodes
    for v in range(g.n):
        x, y = pos[v]
        if v in art_set:
            fill, stroke = "#ff6b6b", "#ffd43b"
        else:
            fill, stroke = "#06d6a0", "#0d1117"
        parts.append(f'<circle cx="{x}" cy="{y}" r="16" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{x}" y="{y + 5}" fill="#0d1117" font-size="14" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    # legend
    parts.append(f'<circle cx="40" cy="{height-30}" r="9" fill="#ff6b6b" stroke="#ffd43b" '
                 f'stroke-width="2"/>')
    parts.append(f'<text x="56" y="{height-25}" fill="#8b949e" font-size="12">articulation point'
                 f'</text>')
    parts.append(f'<line x1="230" y1="{height-30}" x2="270" y2="{height-30}" stroke="#ff6b6b" '
                 f'stroke-width="3.5"/>')
    parts.append(f'<text x="280" y="{height-25}" fill="#8b949e" font-size="12">bridge</text>')
    parts.append("</svg>")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
