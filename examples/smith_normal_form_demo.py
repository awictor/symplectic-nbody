"""Demo: Smith Normal Form diagonalizing an integer matrix and reading off an abelian group.

Reduces an integer matrix to Smith Normal Form, verifies U A V = D and the divisibility chain, and
interprets the invariant factors as the structure of the cokernel abelian group -- the same
computation that yields homology groups in topology. Draws the original matrix and its diagonal SNF as
heatmaps.

    python examples/smith_normal_form_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smith_normal_form import (  # noqa: E402
    smith_normal_form, invariant_factors, abelian_group_structure, is_unimodular, _matmul,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Smith Normal Form: integer diagonalization D = U A V, and the group it encodes\n")

    A = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
    U, D, V = smith_normal_form(A)

    print("  A =")
    for row in A:
        print("      " + " ".join(f"{v:>4}" for v in row))

    print("\n  D = U A V (Smith Normal Form) =")
    for row in D:
        print("      " + " ".join(f"{v:>4}" for v in row))

    check = _matmul(_matmul(U, A), V) == D
    print(f"\n  U A V = D verified: {check}")
    print(f"  U unimodular: {is_unimodular(U)},  V unimodular: {is_unimodular(V)}")

    facs = invariant_factors(D)
    print(f"\n  invariant factors: {facs}  (each divides the next: "
          f"{all(facs[i+1] % facs[i] == 0 for i in range(len(facs)-1))})")

    torsion, free = abelian_group_structure(A)
    parts = [f"Z/{t}" for t in torsion] + (["Z"] * free if free else [])
    group = " x ".join(parts) if parts else "0"
    print(f"\n  cokernel Z^3 / A Z^3 = {group}")
    print(f"  (this is exactly how a homology group H = ker/im is computed: SNF of the")
    print(f"  boundary matrix gives the torsion coefficients and the free rank = Betti number.)")

    # a topology-flavored example: boundary matrix of a triangle (edges->vertices)
    print(f"\n  Topology example: 1-boundary matrix of a filled triangle (3 edges, 3 vertices)")
    # edges: (0,1),(1,2),(2,0); boundary maps edge to v_end - v_start
    boundary = [[-1, 0, 1],
                [1, -1, 0],
                [0, 1, -1]]
    tor, fr = abelian_group_structure(boundary)
    print(f"    invariant factors {invariant_factors(smith_normal_form(boundary)[1])}, "
          f"free rank (H_0 Betti) = {fr}")
    print(f"    -> one connected component (H_0 = Z), no 1-cycles killed incorrectly")

    _svg(os.path.join(outdir, "smith_normal_form.svg"), A, D, facs)
    print(f"\n  wrote {os.path.join(outdir, 'smith_normal_form.svg')}")


def _svg(path, A, D, facs, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Integer matrix A (left) reduced to its diagonal Smith Normal Form D (right)</text>',
    ]

    def draw(mat, ox, oy, label):
        m, n = len(mat), len(mat[0])
        cell = 42
        vmax = max(abs(v) for row in mat for v in row) or 1
        for i in range(m):
            for j in range(n):
                v = mat[i][j]
                t = abs(v) / vmax
                if v == 0:
                    col = "#161b22"
                elif v > 0:
                    col = f"rgb({int(40+40*t)},{int(90+120*t)},{int(180*t+50)})"
                else:
                    col = f"rgb({int(180*t+50)},{int(90-40*t)},{int(60*t+30)})"
                x = ox + j * cell
                y = oy + i * cell
                parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{col}" '
                             f'stroke="#30363d"/>')
                parts.append(f'<text x="{x+cell/2-1:.0f}" y="{y+cell/2+4:.0f}" fill="#e6edf3" '
                             f'font-size="12" text-anchor="middle">{v}</text>')
        parts.append(f'<text x="{ox + n*cell/2:.0f}" y="{oy - 8}" fill="#8b949e" font-size="12" '
                     f'text-anchor="middle">{label}</text>')

    draw(A, 90, 90, "A")
    parts.append(f'<text x="{width/2:.0f}" y="{160}" fill="#ffd43b" font-size="24" '
                 f'text-anchor="middle">-&gt;</text>')
    draw(D, 420, 90, "D = U A V")
    parts.append(f'<text x="{width/2:.0f}" y="{height-24}" fill="#06d6a0" font-size="12" '
                 f'text-anchor="middle">invariant factors: {facs}  ->  cokernel torsion</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
