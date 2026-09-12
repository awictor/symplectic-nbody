"""Demo: the Matrix-Tree theorem -- counting spanning trees with one determinant.

Counts the spanning trees of several graphs via the Laplacian cofactor, confirms Cayley's n^(n-2) for
complete graphs, and draws a small graph beside its Laplacian and the resulting count.

    python examples/matrix_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matrix_tree import (count_spanning_trees, brute_count_spanning_trees,  # noqa: E402
                         weighted_spanning_tree_sum, laplacian)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Matrix-Tree theorem: spanning trees = any cofactor of the Laplacian\n")

    graphs = [
        ("triangle K3", 3, [(0, 1), (1, 2), (2, 0)]),
        ("square C4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("K4", 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
        ("diamond", 4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]),
        ("path P5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ]
    print(f"  {'graph':14s} {'spanning trees':>14}  matches brute")
    for name, n, edges in graphs:
        mt = count_spanning_trees(n, edges)
        bf = brute_count_spanning_trees(n, edges)
        print(f"  {name:14s} {mt:>14}  {mt == bf}")

    print("\n  Cayley's formula falls out for complete graphs K_n = n^(n-2):")
    print(f"    {'n':>2}  {'K_n trees':>10}  {'n^(n-2)':>10}")
    for n in range(2, 8):
        kn = [(i, j) for i in range(n) for j in range(i + 1, n)]
        print(f"    {n:>2}  {count_spanning_trees(n, kn):>10}  {n**(n-2):>10}")

    # weighted example
    a, b, c = 2, 3, 5
    tri = [(0, 1, a), (1, 2, b), (2, 0, c)]
    print(f"\n  weighted triangle (edges {a},{b},{c}): sum over spanning trees of edge-weight products")
    print(f"    = {weighted_spanning_tree_sum(3, tri)}  (= {a}*{b} + {b}*{c} + {c}*{a} "
          f"= {a*b+b*c+c*a})")

    print("\n  Build L = D - A (degrees on the diagonal, -1 for each edge off it), delete any one row")
    print("  and its column, take the determinant -- that integer is the exact spanning-tree count. A")
    print("  single linear-algebra step replaces an exponential search over all edge subsets.")

    _svg(os.path.join(outdir, "matrix_tree.svg"), 4,
         [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    print(f"\n  wrote {os.path.join(outdir, 'matrix_tree.svg')}")


def _svg(path, n, edges, width=720, height=430):
    count = count_spanning_trees(n, edges)
    L = laplacian(n, edges)

    pos = {0: (150, 130), 1: (150, 320), 2: (330, 130), 3: (330, 320)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'A graph, its Laplacian, and its {count} spanning trees</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'delete any row and column of L, take the determinant -> the spanning-tree count</text>',
    ]

    # graph
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4dabf7" '
                     f'stroke-width="2"/>')
    for v, (x, y) in pos.items():
        parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="#06d6a0"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#0d1117" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    # Laplacian matrix on the right
    mx, my = 470, 120
    cell = 42
    parts.append(f'<text x="{mx}" y="{my-16}" fill="#e6edf3" font-size="14">Laplacian L = D - A:</text>')
    for i in range(n):
        for j in range(n):
            x = mx + j * cell
            y = my + i * cell
            val = int(L[i][j])
            col = "#ffd43b" if i == j else ("#ff6b6b" if val < 0 else "#8b949e")
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" '
                         f'fill="#161b22" stroke="#30363d"/>')
            parts.append(f'<text x="{x+(cell-3)/2:.0f}" y="{y+(cell-3)/2+5:.0f}" fill="{col}" '
                         f'font-size="14" text-anchor="middle">{val}</text>')
    parts.append(f'<text x="{mx}" y="{my + n*cell + 22}" fill="#06d6a0" font-size="13">'
                 f'cofactor det = {count} spanning trees</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
