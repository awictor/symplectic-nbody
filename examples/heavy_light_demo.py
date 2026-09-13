"""Demo: heavy-light decomposition -- tree-path queries as contiguous array ranges.

Builds a rooted tree with per-vertex values, decomposes it into heavy chains, answers a few path
sum/max queries (and one after an update), and draws the tree coloured by chain.

    python examples/heavy_light_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heavy_light import HeavyLight  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # a rooted tree:      0
    #                   / | \
    #                  1  2  3
    #                 /|     |
    #                4 5     6
    #               /         \
    #              7           8
    #             /
    #            9
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6), (4, 7), (6, 8), (7, 9)]
    values = [5, 3, 8, 2, 7, 1, 4, 6, 9, 10]
    n = len(values)
    hld = HeavyLight(n, edges, values=values, root=0)

    print("Heavy-light decomposition: path queries in O(log^2 n)\n")
    print(f"  {n}-vertex rooted tree, values {values}\n")

    print("  Decomposition (vertex: heavy-child, chain-head, array-pos):")
    for v in range(n):
        hc = hld.heavy[v]
        print(f"    v{v}: heavy={'-' if hc == -1 else 'v%d' % hc}  head=v{hld.head[v]}  pos={hld.pos[v]}")

    print(f"\n  Max light edges (chain switches) on any root path: "
          f"{hld.max_light_edges_on_any_root_path()}  (the O(log n) bound in action)\n")

    queries = [(9, 8), (7, 5), (9, 2), (4, 6)]
    print(f"  {'path':>10}   {'sum':>5}  {'max':>5}   LCA")
    for u, v in queries:
        print(f"  v{u:>2} -> v{v:<2}   {hld.path_sum(u, v):>5}  {hld.path_max(u, v):>5}   v{hld.lca(u, v)}")

    print("\n  Update v7 from 6 to 50, then re-query the v9->v8 path:")
    hld.update(7, 50)
    print(f"    sum = {hld.path_sum(9, 8)}, max = {hld.path_max(9, 8)}")

    print("\n  Each path splits into O(log n) chain pieces, each a contiguous segment-tree range.")

    _svg(os.path.join(outdir, "heavy_light.svg"), n, edges, hld, values)
    print(f"\n  wrote {os.path.join(outdir, 'heavy_light.svg')}")


def _svg(path, n, edges, hld, values, width=760, height=460):
    # layered layout by depth
    layers = {}
    for v in range(n):
        layers.setdefault(hld.depth[v], []).append(v)
    maxd = max(layers)
    pos = {}
    for d in range(maxd + 1):
        row = sorted(layers[d])
        for j, v in enumerate(row):
            x = width * (j + 1) / (len(row) + 1)
            y = 80 + d * (height - 130) / max(1, maxd)
            pos[v] = (x, y)

    # color each chain distinctly
    palette = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#b197fc", "#ff6b6b", "#e6edf3"]
    heads = sorted(set(hld.head))
    chain_col = {h: palette[i % len(palette)] for i, h in enumerate(heads)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="30" fill="#e6edf3" font-size="15">'
        'Tree coloured by heavy chain (thick = heavy edge, thin = light)</text>',
    ]

    for a, b in edges:
        # child is the deeper one
        child = a if hld.depth[a] > hld.depth[b] else b
        parent = b if child == a else a
        x1, y1 = pos[parent]
        x2, y2 = pos[child]
        heavy = hld.heavy[parent] == child
        col = chain_col[hld.head[child]] if heavy else "#30363d"
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="{col}" stroke-width="{4 if heavy else 1.3}"/>')

    for v in range(n):
        x, y = pos[v]
        col = chain_col[hld.head[v]]
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="17" fill="#161b22" stroke="{col}" '
                     f'stroke-width="2.5"/>')
        parts.append(f'<text x="{x:.0f}" y="{y-1:.0f}" fill="{col}" font-size="9" '
                     f'text-anchor="middle">v{v}</text>')
        parts.append(f'<text x="{x:.0f}" y="{y+9:.0f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{values[v]}</text>')

    parts.append('<text x="20" y="%d" fill="#8b949e" font-size="10">'
                 'each colour is one chain, laid out contiguously in the segment tree</text>'
                 % (height - 15))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
