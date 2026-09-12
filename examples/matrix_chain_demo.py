"""Demo: optimal matrix-chain parenthesization by dynamic programming.

Finds the cheapest order to multiply a chain of matrices, compares to the naive left-to-right order,
and shows the DP cost table. Draws the cost table as a heatmap with the optimal splits.

    python examples/matrix_chain_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matrix_chain import min_cost, left_to_right_cost  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Matrix-chain multiplication: the cheapest order to multiply a chain\n")

    # the classic CLRS example
    dims = [30, 35, 15, 5, 10, 20, 25]
    n = len(dims) - 1
    print(f"  {n} matrices with dimensions:")
    for i in range(1, n + 1):
        print(f"    A{i}: {dims[i-1]} x {dims[i]}")
    cost, parens = min_cost(dims)
    naive = left_to_right_cost(dims)
    print(f"\n  optimal parenthesization: {parens}")
    print(f"    optimal cost:        {cost:,} scalar multiplications")
    print(f"    left-to-right cost:  {naive:,}")
    print(f"    savings: {100*(1-cost/naive):.1f}%")

    # a more skewed example where the gap is bigger
    print("\n  A skewed chain where order matters even more:")
    skewed = [50, 5, 100, 5, 100, 5]
    cost2, p2 = min_cost(skewed)
    naive2 = left_to_right_cost(skewed)
    print(f"    dims {skewed}: optimal {cost2:,} vs left-to-right {naive2:,} "
          f"({100*(1-cost2/naive2):.0f}% saved)")
    print(f"    optimal order: {p2}")

    print("\n  The number of parenthesizations grows as a Catalan number (exponential), but the")
    print("  DP fills an n x n cost table in O(n^3): m[i][j] = min over split k of the cost of the")
    print("  two halves plus multiplying them. Recording each split reconstructs the optimal order.")

    _svg(os.path.join(outdir, "matrix_chain.svg"), dims)
    print(f"\n  wrote {os.path.join(outdir, 'matrix_chain.svg')}")


def _svg(path, dims, width=760, height=430):
    # recompute the DP table to display it
    n = len(dims) - 1
    INF = float("inf")
    m = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = INF
            for k in range(i, j):
                c = m[i][k] + m[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                m[i][j] = min(m[i][j], c)

    cell = min((width - 200) / n, (height - 120) / n)
    ox, oy = 80, 80
    vmax = max(m[i][j] for i in range(1, n + 1) for j in range(i, n + 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Matrix-chain DP cost table m[i][j]</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'm[i][j] = min cost to multiply matrices i..j; brighter = costlier; the corner m[1][n] is the answer</text>',
    ]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            x = ox + (j - 1) * cell
            y = oy + (i - 1) * cell
            t = m[i][j] / vmax if vmax else 0
            r, g, b = int(30 + t * 80), int(60 + t * 120), int(90 + t * 150)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-2:.1f}" height="{cell-2:.1f}" '
                         f'fill="rgb({r},{g},{b})"/>')
            label = str(m[i][j]) if m[i][j] < 100000 else f"{m[i][j]//1000}k"
            parts.append(f'<text x="{x + (cell-2)/2:.1f}" y="{y + (cell-2)/2 + 4:.1f}" '
                         f'fill="#e6edf3" font-size="9" text-anchor="middle">{label}</text>')
    # highlight the answer cell
    xa = ox + (n - 1) * cell
    parts.append(f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{cell-2:.1f}" height="{cell-2:.1f}" '
                 f'fill="none" stroke="#ffd43b" stroke-width="3"/>')
    parts.append(f'<text x="{ox + cell*n + 10:.0f}" y="{oy + cell/2:.0f}" fill="#ffd43b" '
                 f'font-size="12">answer = m[1][{n}] = {m[1][n]:,}</text>')
    # axis labels
    for k in range(1, n + 1):
        parts.append(f'<text x="{ox + (k-1)*cell + cell/2:.0f}" y="{oy-8:.0f}" fill="#8b949e" '
                     f'font-size="11" text-anchor="middle">{k}</text>')
        parts.append(f'<text x="{ox-10:.0f}" y="{oy + (k-1)*cell + cell/2 + 4:.0f}" fill="#8b949e" '
                     f'font-size="11" text-anchor="end">{k}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
