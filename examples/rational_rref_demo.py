"""Demo: exact rational RREF -- rank, null space, and solving without rounding.

Row-reduces matrices over exact fractions, reads off the rank, finds the null space, and solves the
three kinds of linear system (unique, none, infinite). Draws a matrix's RREF as a colour grid marking
pivot vs free columns.

    python examples/rational_rref_demo.py [output_dir]
"""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rational_rref import rref, rank, null_space, solve, matvec, is_zero_vector  # noqa: E402


def fmt(x):
    return str(x)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Exact rational RREF: linear algebra with certainty, no floating-point error\n")

    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    R, pivots = rref(A)
    print("  A =")
    for row in A:
        print("    " + "  ".join(f"{v:3}" for v in row))
    print("\n  RREF(A) =")
    for row in R:
        print("    " + "  ".join(f"{fmt(v):>6}" for v in row))
    print(f"\n  pivot columns: {pivots}   rank = {rank(A)}")
    ns = null_space(A)
    print(f"  null space (dim {len(ns)}): " +
          ", ".join("(" + ", ".join(fmt(x) for x in v) + ")" for v in ns))
    print(f"  null-space vector satisfies A x = 0: {all(is_zero_vector(matvec(A, v)) for v in ns)}")

    print("\n  the three kinds of linear system:")
    kind, sol = solve([[2, 1], [1, 3]], [3, 5])
    print(f"    unique  : 2x+y=3, x+3y=5 -> x = ({fmt(sol[0])}, {fmt(sol[1])})")
    kind, _ = solve([[1, 1], [1, 1]], [1, 2])
    print(f"    none    : x+y=1, x+y=2 -> {kind} (inconsistent)")
    kind, data = solve([[1, 1, 1]], [6])
    x0, basis = data
    print(f"    infinite: x+y+z=6 -> particular ({', '.join(fmt(v) for v in x0)}) + "
          f"{len(basis)} free directions")

    print("\n  Working over Python's Fraction means every pivot test is exact -- so 'is this pivot")
    print("  really zero?' is answered with certainty and the rank is never misjudged, unlike a")
    print("  floating-point solver that might see 1e-16 and guess wrong. Essential for computer")
    print("  algebra, exact geometry, and any rank decision that must be provably correct.")

    _svg(os.path.join(outdir, "rational_rref.svg"), A)
    print(f"\n  wrote {os.path.join(outdir, 'rational_rref.svg')}")


def _svg(path, A, cell=54):
    R, pivots = rref(A)
    rows = len(R)
    cols = len(R[0])
    pivot_set = set(pivots)
    ox, oy = 60, 90
    width = ox * 2 + cols * cell
    height = oy + rows * cell + 60

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="17">'
        f'RREF over exact fractions: rank {len(pivots)} of {cols} columns</text>',
        f'<text x="20" y="54" fill="#8b949e" font-size="12">'
        f'green columns are pivot (basic) columns; grey are free -- one null-space vector each</text>',
    ]
    # column headers
    for c in range(cols):
        x = ox + c * cell
        col = "#06d6a0" if c in pivot_set else "#8b949e"
        label = "pivot" if c in pivot_set else "free"
        parts.append(f'<text x="{x+cell/2:.0f}" y="{oy-8}" fill="{col}" font-size="10" '
                     f'text-anchor="middle">{label}</text>')
    for i in range(rows):
        for j in range(cols):
            x = ox + j * cell
            y = oy + i * cell
            v = R[i][j]
            bg = "#132a1f" if j in pivot_set else "#161b22"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{bg}" '
                         f'stroke="#30363d"/>')
            txt = "#06d6a0" if (j in pivot_set and v != 0) else "#e6edf3"
            parts.append(f'<text x="{x+(cell-2)/2:.0f}" y="{y+(cell-2)/2+4:.0f}" fill="{txt}" '
                         f'font-size="13" text-anchor="middle">{str(v)}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
