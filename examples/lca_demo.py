"""Demo: lowest common ancestor by binary lifting.

Builds a small rooted tree, answers a few LCA / distance / k-th-ancestor queries, and draws the
tree with one query pair and their LCA highlighted.

    python examples/lca_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lca import LCA  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # A rooted tree (a mock org chart / taxonomy):
    #             0
    #          /  |  \
    #         1   2   3
    #        / \      |
    #       4   5     6
    #      /|         |
    #     7 8         9
    names = ["CEO", "VP-Eng", "VP-Sales", "VP-Ops", "Dir-Backend", "Dir-Frontend",
             "Dir-Field", "Mgr-API", "Mgr-Data", "Mgr-West"]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6), (4, 7), (4, 8), (6, 9)]
    n = len(names)
    tree = LCA(n, edges=edges, root=0)

    print("Lowest common ancestor by binary lifting: O(log n) per query\n")
    print("  A 10-node org tree rooted at CEO. Precomputed 2^k-ancestor table, then:\n")

    queries = [(7, 8), (7, 5), (8, 9), (7, 9), (5, 2)]
    print(f"  {'u':>12}  {'v':>12}   {'LCA':>10}  {'dist':>4}")
    for u, v in queries:
        w = tree.lca(u, v)
        print(f"  {names[u]:>12}  {names[v]:>12}   {names[w]:>10}  {tree.distance(u, v):>4}")

    print("\n  k-th ancestor of Mgr-API (node 7):")
    for k in range(0, 4):
        a = tree.kth_ancestor(7, k)
        print(f"    {k} up -> {names[a] if a != -1 else '(above root)'}")

    print("\n  Path from Mgr-API to Mgr-West (jump one step at a time):")
    dist = tree.distance(7, 9)
    path = [names[tree.jump(7, 9, k)] for k in range(dist + 1)]
    print("    " + " -> ".join(path))
    print("\n  dist(u,v) = depth[u] + depth[v] - 2*depth[lca]. The whole path routes through the LCA.")

    _svg(os.path.join(outdir, "lca.svg"), names, edges, tree, hi_u=7, hi_v=9)
    print(f"\n  wrote {os.path.join(outdir, 'lca.svg')}")


def _svg(path, names, edges, tree, hi_u, hi_v, width=760, height=440):
    n = len(names)
    # layered layout by depth; assign x within each layer by discovery order
    children = {i: [] for i in range(n)}
    for a, b in edges:
        children[a].append(b)
    layers = {}
    for v in range(n):
        layers.setdefault(tree.depth[v], []).append(v)
    maxd = max(layers)
    pos = {}
    for d in range(maxd + 1):
        row = layers[d]
        m = len(row)
        for j, v in enumerate(sorted(row)):
            x = (width) * (j + 1) / (m + 1)
            y = 70 + d * (height - 110) / max(1, maxd)
            pos[v] = (x, y)

    w = tree.lca(hi_u, hi_v)
    # the highlighted path (u up to lca, then down to v)
    dist = tree.distance(hi_u, hi_v)
    path_nodes = {tree.jump(hi_u, hi_v, k) for k in range(dist + 1)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="15">'
        f'LCA({names[hi_u]}, {names[hi_v]}) = {names[w]}  '
        f'(path highlighted, routes through the LCA)</text>',
    ]

    def on_path_edge(a, b):
        return a in path_nodes and b in path_nodes

    for a, b in edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        hot = on_path_edge(a, b)
        col = "#ffd43b" if hot else "#30363d"
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="{col}" stroke-width="{3 if hot else 1.5}"/>')

    for v in range(n):
        cx, cy = pos[v]
        if v == w:
            col, r = "#ff6b6b", 18       # the LCA
        elif v in (hi_u, hi_v):
            col, r = "#06d6a0", 16       # the query endpoints
        elif v in path_nodes:
            col, r = "#ffd43b", 14       # on the path
        else:
            col, r = "#4dabf7", 13
        parts.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}" fill="#161b22" '
                     f'stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{cx:.0f}" y="{cy+3:.0f}" fill="{col}" font-size="8" '
                     f'text-anchor="middle">{v}</text>')
        parts.append(f'<text x="{cx:.0f}" y="{cy-r-4:.0f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{names[v]}</text>')

    parts.append('<text x="20" y="%d" fill="#8b949e" font-size="10">'
                 'red = LCA, green = query nodes, gold = path</text>' % (height - 15))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
