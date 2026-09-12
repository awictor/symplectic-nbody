"""Demo: Prufer sequences -- labeled trees as integer strings, and Cayley's formula.

Encodes a few labeled trees to their Prufer sequences and back, shows how the degree sequence is read
straight off the code, and confirms Cayley's n^(n-2) count by exhaustive enumeration. Draws a tree
beside its Prufer sequence.

    python examples/prufer_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prufer import (tree_to_prufer, prufer_to_tree, cayley_count,  # noqa: E402
                    degree_from_prufer, all_labeled_trees)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Prufer sequences: a bijection between labeled trees and integer strings\n")

    trees = [
        ("star (hub 0)", 6, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]),
        ("path 0..5", 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        ("caterpillar", 6, [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)]),
    ]
    for name, n, edges in trees:
        seq = tree_to_prufer(n, edges)
        back = prufer_to_tree(n, seq)
        deg = degree_from_prufer(n, seq)
        print(f"  {name:14s} Prufer {seq}  degrees {deg}  roundtrip {'ok' if back == sorted((min(u,v),max(u,v)) for u,v in edges) else 'FAIL'}")

    print("\n  A vertex appears in the sequence (its degree - 1) times, so leaves never appear --")
    print("  read the tree's whole degree sequence straight off the code.\n")

    print("  Cayley's formula: exactly n^(n-2) labeled trees on n vertices.")
    print(f"    {'n':>2}  {'n^(n-2)':>10}  {'enumerated':>10}")
    for n in range(1, 8):
        enumerated = len(set(all_labeled_trees(n))) if n <= 7 else None
        print(f"    {n:>2}  {cayley_count(n):>10}  {enumerated:>10}")
    print("\n  Every length-(n-2) string over {0..n-1} is a valid Prufer sequence, and each gives a")
    print("  distinct tree -- so counting trees reduces to counting strings, and Cayley's formula")
    print("  falls out immediately. Decoding a uniformly random string yields a uniformly random tree.")

    _svg(os.path.join(outdir, "prufer.svg"), 8, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (2, 6), (6, 7)])
    print(f"\n  wrote {os.path.join(outdir, 'prufer.svg')}")


def _svg(path, n, edges, width=760, height=420):
    seq = tree_to_prufer(n, edges)
    deg = degree_from_prufer(n, seq)

    # layered layout rooted at 0
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    depth = {0: 0}
    parent = {0: -1}
    order = [0]
    seen = {0}
    i = 0
    while i < len(order):
        u = order[i]
        i += 1
        for w in sorted(adj[u]):
            if w not in seen:
                seen.add(w)
                depth[w] = depth[u] + 1
                parent[w] = u
                order.append(w)
    by_depth = {}
    for v in range(n):
        by_depth.setdefault(depth[v], []).append(v)
    pos = {}
    maxd = max(depth.values())
    for d, vs in by_depth.items():
        for j, v in enumerate(vs):
            x = 80 + (width - 300) * (j + 1) / (len(vs) + 1)
            y = 110 + (height - 200) * (d / max(1, maxd))
            pos[v] = (x, y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'A labeled tree and its Prufer sequence</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'leaves (yellow) never appear in the code; each vertex appears (degree - 1) times</text>',
    ]

    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#484f58" stroke-width="1.5"/>')
    for v, (x, y) in pos.items():
        is_leaf = deg[v] == 1
        fill = "#ffd43b" if is_leaf else "#4dabf7"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="17" fill="{fill}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#0d1117" font-size="13" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    # the Prufer sequence shown as boxes on the right
    bx = width - 150
    parts.append(f'<text x="{bx}" y="100" fill="#e6edf3" font-size="13">Prufer:</text>')
    for i, x in enumerate(seq):
        yy = 120 + i * 34
        parts.append(f'<rect x="{bx}" y="{yy}" width="30" height="28" fill="#161b22" '
                     f'stroke="#4dabf7"/>')
        parts.append(f'<text x="{bx+15}" y="{yy+19}" fill="#e6edf3" font-size="14" '
                     f'text-anchor="middle">{x}</text>')
    parts.append(f'<text x="{bx}" y="{120 + len(seq)*34 + 20}" fill="#8b949e" font-size="11">'
                 f'length {len(seq)} = n-2</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
