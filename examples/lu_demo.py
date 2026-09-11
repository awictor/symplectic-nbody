"""Demo: LU and Cholesky decomposition.

Factors a matrix as P A = L U and a symmetric positive-definite one as A = L L', showing the
triangular factors reconstruct the original, that the determinant is the product of U's diagonal,
that solving reuses one factorization for many right-hand sides, and that Cholesky doubles as the
positive-definiteness test. Draws the L, U, and L-L' factor matrices as shaded grids.

    python examples/lu_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lu import (lu_decompose, lu_solve, determinant, inverse,  # noqa: E402
                cholesky, cholesky_solve, is_positive_definite)


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _fmt(M):
    return "\n".join("    [" + "  ".join(f"{v:6.2f}" for v in row) + "]" for row in M)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    A = [[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]]
    L, U, piv, sign = lu_decompose(A)

    print("LU decomposition: P A = L U by Gaussian elimination with partial pivoting\n")
    print("  A =")
    print(_fmt(A))
    print("\n  L (unit lower triangular) =")
    print(_fmt(L))
    print("\n  U (upper triangular) =")
    print(_fmt(U))
    print(f"\n  pivot row order {piv}, permutation sign {int(sign):+d}")
    PA = [[A[piv[i]][j] for j in range(3)] for i in range(3)]
    err = max(abs(PA[i][j] - _matmul(L, U)[i][j]) for i in range(3) for j in range(3))
    print(f"  reconstruction max|PA - LU| = {err:.2e}\n")

    detU = sign
    for i in range(3):
        detU *= U[i][i]
    print(f"  det(A) = sign * prod(diag U) = {detU:.1f}  (== determinant() {determinant(A):.1f})\n")

    print("  Factor once, then solve for many right-hand sides cheaply (O(n^2) each):")
    for b in ([5.0, -2.0, 9.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]):
        x = lu_solve(A, b)
        chk = [round(sum(A[i][j] * x[j] for j in range(3)), 6) for i in range(3)]
        print(f"    A x = {b}  ->  x = [{', '.join(f'{v:.3f}' for v in x)}]  (A x back = {chk})")

    # Cholesky on an SPD matrix
    S = [[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]]
    Lc = cholesky(S)
    print("\nCholesky: A = L L' for a symmetric positive-definite A (half the work, no pivoting)\n")
    print("  A =")
    print(_fmt(S))
    print("\n  L =")
    print(_fmt(Lc))
    Lt = [[Lc[j][i] for j in range(3)] for i in range(3)]
    cerr = max(abs(_matmul(Lc, Lt)[i][j] - S[i][j]) for i in range(3) for j in range(3))
    print(f"\n  reconstruction max|L L' - A| = {cerr:.2e}")

    print("\n  Attempting Cholesky IS the positive-definiteness test:")
    for name, M in [("SPD [[4,2,2],...]", S),
                    ("indefinite [[1,2],[2,1]]", [[1.0, 2.0], [2.0, 1.0]]),
                    ("negative [[-1,0],[0,1]]", [[-1.0, 0.0], [0.0, 1.0]])]:
        ok = is_positive_definite(M)
        print(f"    {name:>26}: positive definite = {ok}")

    print("\n  LU factors any square matrix and powers solves, determinants, and inverses;")
    print("  Cholesky is the smaller, stabler SPD special case behind least squares and Kalman.")

    _svg(os.path.join(outdir, "lu.svg"), A, L, U, S, Lc)
    print(f"\n  wrote {os.path.join(outdir, 'lu.svg')}")


def _svg(path, A, L, U, S, Lc, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'LU and Cholesky: a matrix as a product of triangular factors</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'cells shaded by magnitude (blue negative, red positive); zeros show the triangular '
        f'structure</text>',
    ]

    def grid(M, x0, y0, cell, title, sub):
        n = len(M)
        vmax = max((abs(M[i][j]) for i in range(n) for j in range(len(M[0]))), default=1.0) or 1.0
        parts.append(f'<text x="{x0 + n*cell/2:.1f}" y="{y0-14:.1f}" fill="#e6edf3" '
                     f'font-size="13" text-anchor="middle">{title}</text>')
        parts.append(f'<text x="{x0 + n*cell/2:.1f}" y="{y0-2:.1f}" fill="#8b949e" '
                     f'font-size="9" text-anchor="middle">{sub}</text>')
        for i in range(n):
            for j in range(len(M[0])):
                v = M[i][j]
                t = abs(v) / vmax
                if abs(v) < 1e-12:
                    col = "#161b22"
                elif v > 0:
                    col = f"rgb({int(40+t*215)},{int(50+t*50)},{int(60)})"
                else:
                    col = f"rgb({int(40)},{int(70+t*60)},{int(60+t*195)})"
                cx = x0 + j * cell
                cy = y0 + i * cell
                parts.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell-1.5:.1f}" '
                             f'height="{cell-1.5:.1f}" fill="{col}"/>')
                parts.append(f'<text x="{cx+cell/2:.1f}" y="{cy+cell/2+3:.1f}" fill="#e6edf3" '
                             f'font-size="9" text-anchor="middle">{v:.1f}</text>')

    c = 44
    # top row: A = L U
    grid(A, 45, 90, c, "A", "original")
    grid(L, 250, 90, c, "L", "unit lower")
    grid(U, 455, 90, c, "U", "upper")
    parts.append(f'<text x="222" y="{90+1.4*c:.1f}" fill="#8b949e" font-size="20">=</text>')
    parts.append(f'<text x="428" y="{90+1.4*c:.1f}" fill="#8b949e" font-size="20">x</text>')

    # bottom row: S = L_c L_c'
    y2 = 290
    grid(S, 45, y2, c, "A (SPD)", "symmetric pos-def")
    grid(Lc, 250, y2, c, "L", "Cholesky")
    Lt = [[Lc[j][i] for j in range(3)] for i in range(3)]
    grid(Lt, 455, y2, c, "L'", "transpose")
    parts.append(f'<text x="222" y="{y2+1.4*c:.1f}" fill="#8b949e" font-size="20">=</text>')
    parts.append(f'<text x="428" y="{y2+1.4*c:.1f}" fill="#8b949e" font-size="20">x</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
