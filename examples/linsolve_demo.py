"""Demo: Gaussian elimination and LU decomposition -- solving A x = b.

Solves a linear system, shows the LU factors satisfying P A = L U, reuses one factorization for
several right-hand sides, and reads the determinant and inverse off the factorization. Draws the
L and U triangular factors as heatmaps.

    python examples/linsolve_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linsolve import (solve, lu_decompose, solve_lu, determinant,  # noqa: E402
                      inverse, matmul, matvec, residual_norm)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    print("Gaussian elimination / LU: solve A x = b, factor once, reuse\n")
    print("  A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],  b = [8, -11, -3]")
    x = solve(A, b)
    print(f"  solution x = {[round(v, 6) for v in x]}   residual ||Ax-b|| = {residual_norm(A, x, b):.1e}\n")

    L, U, perm, sign = lu_decompose(A)
    print(f"  LU factorization (P A = L U), row permutation {perm}:")
    print("    L (unit lower):", [[round(v, 3) for v in row] for row in L])
    print("    U (upper):     ", [[round(v, 3) for v in row] for row in U])
    PA = [A[perm[i]] for i in range(3)]
    print(f"    check P A == L U: {all(abs(PA[i][j] - matmul(L, U)[i][j]) < 1e-9 for i in range(3) for j in range(3))}\n")

    print("  Reusing the factorization for several right-hand sides (each O(n^2)):")
    for rhs in ([8, -11, -3], [1, 0, 0], [0, 5, 5]):
        print(f"    b = {rhs}  ->  x = {[round(v, 4) for v in solve_lu(L, U, perm, rhs)]}")

    print(f"\n  determinant = {determinant(A):.6g}  (sign {sign} x product of U's diagonal)")
    inv = inverse(A)
    print(f"  inverse row 0 = {[round(v, 4) for v in inv[0]]}")
    print(f"  A * inverse = identity: "
          f"{all(abs(matmul(A, inv)[i][j] - (1 if i == j else 0)) < 1e-9 for i in range(3) for j in range(3))}")
    print("\n  Factor once in O(n^3), then every new b costs O(n^2): the reason LU beats")
    print("  re-eliminating from scratch. Partial pivoting (swap in the largest pivot) keeps it")
    print("  numerically stable -- it underlies circuit analysis, FEM, and every linear solve.")

    _svg(os.path.join(outdir, "linsolve.svg"), L, U)
    print(f"\n  wrote {os.path.join(outdir, 'linsolve.svg')}")


def _svg(path, L, U, w=760, h=360):
    n = len(L)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'LU decomposition: A splits into triangular L and U</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'L unit-lower-triangular (left), U upper-triangular (right); '
        f'together they factor the pivoted A</text>',
    ]

    cell = 56

    def draw(mat, x0, title, color):
        vmax = max(abs(mat[i][j]) for i in range(n) for j in range(n)) or 1
        y0 = 90
        parts.append(f'<text x="{x0 + n * cell / 2:.1f}" y="{y0 - 12:.1f}" fill="#e6edf3" '
                     f'font-size="13" text-anchor="middle">{title}</text>')
        for i in range(n):
            for j in range(n):
                x, y = x0 + j * cell, y0 + i * cell
                v = mat[i][j]
                if v == 0:
                    fill = "#161b22"
                else:
                    t = abs(v) / vmax
                    fill = color(t)
                parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                             f'fill="{fill}" stroke="#0d1117" stroke-width="1"/>')
                parts.append(f'<text x="{x + cell/2 - 1:.1f}" y="{y + cell/2 + 4:.1f}" '
                             f'fill="{"#8b949e" if v == 0 else "#e6edf3"}" font-size="11" '
                             f'text-anchor="middle">{v:.2f}</text>')

    def blue(t):
        return f"#{int(0x16 + t*(0x4d-0x16)):02x}{int(0x1b + t*(0xab-0x1b)):02x}{int(0x22 + t*(0xf7-0x22)):02x}"

    def green(t):
        return f"#{int(0x16 + t*(0x06-0x16)+t*0):02x}{int(0x1b + t*(0xd6-0x1b)):02x}{int(0x22 + t*(0x6a-0x22)):02x}"

    draw(L, 60, "L", blue)
    draw(U, 60 + n * cell + 80, "U", green)

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
