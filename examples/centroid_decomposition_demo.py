"""Demo: centroid decomposition -- balanced divide-and-conquer on a tree.

Builds the centroid tree of a sample tree, shows its logarithmic depth, and counts vertex pairs within
a distance threshold via the centroid trick, verifying against brute force. Draws the original tree
with each node tinted by its depth in the centroid tree.

    python examples/centroid_decomposition_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from centroid_decomposition import (build_centroid_tree, count_pairs_within_distance,  # noqa: E402
                                    centroid_tree_depth, brute_count_pairs_within_distance)

# a sample tree: a central path with branches
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # backbone
    (1, 5), (1, 6),                           # branch off 1
    (3, 7), (7, 8),                           # branch off 3
    (2, 9),                                   # branch off 2
]
N = 10


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    parent, root = build_centroid_tree(N, EDGES)
    depth = centroid_tree_depth(N, EDGES)

    print("Centroid decomposition: log-deep divide-and-conquer on a tree\n")
    print(f"  {N}-vertex tree, {len(EDGES)} edges\n")
    print(f"  centroid tree root: {root}")
    print(f"  centroid tree depth: {depth}  (log2({N}) is about {math.log2(N):.1f})\n")

    print("  centroid tree parent of each vertex:")
    print(f"    {parent}")

    print("\n  vertex pairs within distance k (via centroid decomposition):")
    print(f"    {'k':>3}  {'pairs':>6}  matches brute")
    for k in range(1, 6):
        cd = count_pairs_within_distance(N, EDGES, k)
        bf = brute_count_pairs_within_distance(N, EDGES, k)
        print(f"    {k:>3}  {cd:>6}  {cd == bf}")

    print("\n  At each centroid, paths through it are counted by gathering distances to every node and")
    print("  two-pointer counting the pairs summing to <= k, then subtracting per-branch pairs that")
    print("  don't actually cross the centroid. Because each level halves the components, only")
    print("  O(log n) levels exist -- turning an O(n^2) all-pairs question into O(n log^2 n).")

    _svg(os.path.join(outdir, "centroid_decomposition.svg"), parent, root, depth)
    print(f"\n  wrote {os.path.join(outdir, 'centroid_decomposition.svg')}")


_DEPTH_COLORS = ["#ff6b6b", "#ff922b", "#ffd43b", "#06d6a0", "#4dabf7", "#b197fc"]


def _svg(path, parent, root, max_depth, width=760, height=440):
    # depth of each vertex in the centroid tree
    cdepth = [0] * N
    for v in range(N):
        u, d = v, 0
        while parent[u] != -1:
            u = parent[u]
            d += 1
        cdepth[v] = d

    # layout the ORIGINAL tree with a simple BFS from vertex 0
    adj = [[] for _ in range(N)]
    for u, v in EDGES:
        adj[u].append(v)
        adj[v].append(u)
    depth_of = {0: 0}
    par = {0: -1}
    order = [0]
    seen = {0}
    i = 0
    while i < len(order):
        u = order[i]
        i += 1
        for w in sorted(adj[u]):
            if w not in seen:
                seen.add(w)
                depth_of[w] = depth_of[u] + 1
                par[w] = u
                order.append(w)
    by_depth = {}
    for v in range(N):
        by_depth.setdefault(depth_of[v], []).append(v)
    pos = {}
    maxd = max(depth_of.values())
    for d, vs in by_depth.items():
        for j, v in enumerate(vs):
            x = 90 + (width - 180) * (j + 1) / (len(vs) + 1)
            y = 110 + (height - 200) * (d / max(1, maxd))
            pos[v] = (x, y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'The tree, coloured by depth in its centroid decomposition</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'red = the overall centroid (root of the centroid tree); deeper colours split smaller pieces'
        f'</text>',
    ]

    for u, v in EDGES:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#484f58" stroke-width="1.5"/>')
    for v, (x, y) in pos.items():
        col = _DEPTH_COLORS[cdepth[v] % len(_DEPTH_COLORS)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="18" fill="{col}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#0d1117" font-size="12" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    # legend
    for d in range(max_depth + 1):
        yy = 90 + d * 22
        parts.append(f'<circle cx="30" cy="{yy}" r="8" fill="{_DEPTH_COLORS[d % len(_DEPTH_COLORS)]}"/>')
        parts.append(f'<text x="45" y="{yy+4}" fill="#8b949e" font-size="11">level {d}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
