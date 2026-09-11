"""Demo: sparse table O(1) range minimum, and binary-lifting LCA on a tree.

Shows RMQ answering subrange minima in two lookups, then builds a tree and answers lowest-common-
ancestor and distance queries. Draws the array with a highlighted query range and its minimum, and
the tree with a highlighted LCA.

    python examples/sparse_table_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sparse_table import RMQ, SparseTable, LCA  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sparse table: O(1) range minimum, and binary-lifting LCA\n")

    a = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0, 11, 2]
    rmq = RMQ(a)
    print(f"  array: {a}")
    print("  Range minimum queries (each answered by exactly two table lookups):")
    for l, r in [(1, 5), (3, 9), (0, len(a)), (6, 8)]:
        print(f"    min[{l},{r}) = {rmq.min_range(l, r)}   (brute {min(a[l:r])})")

    mx = SparseTable(a, op=max)
    print(f"\n  Same table idea for max: max[3,9) = {mx.query(3, 9)}   (brute {max(a[3:9])})")

    # tree LCA
    # a small binary-ish tree:
    #            0
    #          /   \
    #         1     2
    #        / \   / \
    #       3   4 5   6
    #      /       \
    #     7         8
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (5, 8)]
    lca = LCA(9, edges)
    print("\n  Lowest common ancestor (binary lifting), on a 9-node tree:")
    for u, v in [(7, 4), (7, 8), (3, 4), (8, 6)]:
        print(f"    LCA({u}, {v}) = {lca.query(u, v)}, distance = {lca.distance(u, v)}")

    print("\n  RMQ works because min is idempotent: any range is covered by two overlapping")
    print("  power-of-two blocks, so a query is min(table[l][k], table[r-2^k][k]) -- always O(1).")
    print("  LCA lifts the deeper node to the other's depth, then both jump up by shrinking")
    print("  powers of two until their parents meet -- O(log n) using the 2^k-ancestor table.")

    _svg(os.path.join(outdir, "sparse_table.svg"), a, rmq, edges, lca)
    print(f"\n  wrote {os.path.join(outdir, 'sparse_table.svg')}")


def _svg(path, a, rmq, edges, lca, width=760, height=470):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Sparse table RMQ (top) and binary-lifting LCA (bottom)</text>',
    ]

    # --- top: array bars with a highlighted query range ---
    l, r = 3, 9
    m = rmq.min_range(l, r)
    n = len(a)
    bar_w = (width - 80) / n
    base_y = 200
    amax = max(a)
    parts.append(f'<text x="20" y="55" fill="#8b949e" font-size="12">'
                 f'array with min[{l},{r}) = {m} highlighted (green range, yellow minimum)</text>')
    for i, val in enumerate(a):
        x = 40 + i * bar_w
        h = 10 + val / amax * 110
        in_range = l <= i < r
        is_min = in_range and val == m
        col = "#ffd43b" if is_min else ("#06d6a0" if in_range else "#30363d")
        parts.append(f'<rect x="{x:.1f}" y="{base_y - h:.1f}" width="{bar_w-3:.1f}" height="{h:.1f}" '
                     f'fill="{col}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{base_y + 14:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="middle">{val}</text>')

    # --- bottom: tree with a highlighted LCA ---
    # simple layered layout by depth
    n_nodes = 9
    children = {i: [] for i in range(n_nodes)}
    parent = {0: None}
    for u, v in edges:
        # root at 0, so u is parent of v in our construction order
        children[u].append(v)
        parent[v] = u
    depth = {0: 0}
    order = [0]
    while order:
        u = order.pop()
        for c in children[u]:
            depth[c] = depth[u] + 1
            order.append(c)
    max_depth = max(depth.values())
    # x by an in-order-ish index within each depth
    by_depth = {}
    for v in range(n_nodes):
        by_depth.setdefault(depth[v], []).append(v)
    pos = {}
    top = 280
    for d, nodes in by_depth.items():
        for idx, v in enumerate(sorted(nodes)):
            x = (width) * (idx + 0.5) / len(nodes)
            y = top + d * ((height - top - 30) / (max_depth + 1))
            pos[v] = (x, y)

    u_q, v_q = 7, 8
    anc = lca.query(u_q, v_q)
    parts.append(f'<text x="20" y="{top-15}" fill="#8b949e" font-size="12">'
                 f'tree: LCA({u_q},{v_q}) = {anc} (red), query nodes in blue</text>')
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     f'stroke="#484f58" stroke-width="1.5"/>')
    for v, (x, y) in pos.items():
        if v == anc:
            col = "#ff6b6b"
        elif v in (u_q, v_q):
            col = "#4dabf7"
        else:
            col = "#30363d"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#e6edf3" font-size="13" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
