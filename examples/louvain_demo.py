"""Demo: Louvain community detection on a network with planted communities.

Builds a network of dense groups joined by sparse bridges, runs Louvain, confirms it recovers the
groups and reports the modularity, and draws the graph with each community in its own color.

    python examples/louvain_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from louvain import detect, modularity  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Louvain community detection: finding a network's natural clusters\n")

    # four dense groups of 5 nodes, sparse bridges between them
    groups = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20))]
    n = 20
    edges = []
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                edges.append((g[i], g[j], 1.0))
    bridges = [(4, 5, 1.0), (9, 10, 1.0), (14, 15, 1.0), (19, 0, 1.0)]
    edges += bridges

    comm, q = detect(n, edges, seed=1)
    k = len(set(comm))
    print(f"  {n} nodes, {len(edges)} edges (4 planted groups + 4 bridges)")
    print(f"  Louvain found {k} communities, modularity {q:.4f}")
    for gi, g in enumerate(groups):
        labels = set(comm[node] for node in g)
        print(f"    planted group {gi} ({g}): community label(s) {labels}"
              f"{'  <- recovered' if len(labels) == 1 else ''}")

    print(f"\n  Modularity of the trivial partitions (for comparison):")
    print(f"    all in one community: {modularity(n, edges, [0]*n):.4f}")
    print(f"    every node its own:   {modularity(n, edges, list(range(n))):.4f}")
    print(f"    Louvain:              {q:.4f}  <- highest")

    print("\n  Louvain greedily moves each node to the neighbouring community that most increases")
    print("  modularity, then collapses communities into super-nodes and recurses. Each phase only")
    print("  raises modularity, so it converges fast to a strong hierarchical partition.")

    _svg(os.path.join(outdir, "louvain.svg"), n, edges, comm)
    print(f"\n  wrote {os.path.join(outdir, 'louvain.svg')}")


def _svg(path, n, edges, comm, width=760, height=440):
    colors = ["#4dabf7", "#ffd43b", "#ff6b6b", "#06d6a0", "#b197fc", "#ff922b"]

    # layout: place each community's nodes in a cluster around a ring of community centers
    k = len(set(comm))
    comm_nodes = {c: [i for i in range(n) if comm[i] == c] for c in set(comm)}
    cx, cy = width / 2, height / 2 + 20
    R = 150
    pos = {}
    for ci, c in enumerate(sorted(comm_nodes)):
        ang = 2 * math.pi * ci / k
        gx = cx + R * math.cos(ang)
        gy = cy + R * math.sin(ang)
        members = comm_nodes[c]
        for mi, node in enumerate(members):
            a = 2 * math.pi * mi / max(len(members), 1)
            pos[node] = (gx + 40 * math.cos(a), gy + 40 * math.sin(a))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Louvain communities: {len(set(comm))} clusters found by modularity</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each colour is a community; dense within-group edges, thin bridges between -- the natural clustering</text>',
    ]
    for u, v, w in edges:
        cross = comm[u] != comm[v]
        col = "#ff6b6b" if cross else "#30363d"
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     f'stroke="{col}" stroke-width="{1.6 if cross else 0.9}"/>')
    for node in range(n):
        x, y = pos[node]
        col = colors[comm[node] % len(colors)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="1.5"/>')
    parts.append(f'<text x="20" y="{height-16}" fill="#8b949e" font-size="12">'
                 f'red edges bridge communities; gray edges stay within one</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
