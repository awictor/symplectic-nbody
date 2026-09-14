"""Demo: the resultant detecting common roots, the discriminant flagging repeated roots, elimination.

Shows the resultant vanishing exactly when two polynomials share a root, the discriminant generalizing
b^2 - 4ac to classify a polynomial's roots, and resultant-based elimination solving a two-variable
system (a circle meeting a line). Draws the Sylvester matrix and the discriminant sign regions.

    python examples/resultant_demo.py [output_dir]
"""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resultant import (  # noqa: E402
    resultant, discriminant, has_common_root, has_repeated_root, sylvester_matrix,
)
from durand_kerner import from_roots  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The resultant: detecting common roots without finding them\n")

    print(f"  Res(p, q) = 0 iff p and q share a root:")
    cases = [
        ("(x-1)(x-2)", [1, -3, 2], "(x-2)(x-3)", [1, -5, 6]),
        ("(x-1)(x-2)", [1, -3, 2], "(x-3)(x-4)", [1, -7, 12]),
        ("x^2-2", [1, 0, -2], "x^2-2", [1, 0, -2]),
    ]
    for pn, p, qn, q in cases:
        R = resultant(p, q)
        print(f"    Res[{pn}, {qn}] = {R}  ->  {'COMMON ROOT' if R == 0 else 'no common root'}")

    print(f"\n  The discriminant generalizes b^2 - 4ac to any degree:")
    print(f"    {'polynomial':<22}{'discriminant':>14}{'meaning':>22}")
    disc_cases = [
        ("x^2 - 5x + 6", [1, -5, 6], "two real roots (2,3)"),
        ("x^2 + 1", [1, 0, 1], "complex pair"),
        ("(x-1)^2", [1, -2, 1], "double root"),
        ("x^3 - x", [1, 0, -1, 0], "three real roots"),
        ("x^3 + x + 1", [1, 0, 1, 1], "one real, two complex"),
    ]
    for name, coeffs, meaning in disc_cases:
        d = discriminant(coeffs)
        print(f"    {name:<22}{str(d):>14}{meaning:>22}")

    print(f"\n  Elimination: circle x^2+y^2=1 meets line y=x.")
    print(f"  As polynomials in y: p = y^2 + (x^2-1), q = y - x.")
    print(f"  Res_y(p, q) eliminates y, leaving a polynomial in x whose roots are the x-coordinates:")
    for xval in [0, 1, Fraction(1, 2)]:
        p = [Fraction(1), Fraction(0), Fraction(xval) ** 2 - 1]
        q = [Fraction(1), -Fraction(xval)]
        r = resultant(p, q)
        print(f"    x = {xval}:  Res_y = {r}   (this is 2x^2 - 1; zero at x = +/- 1/sqrt(2))")
    print(f"  So the intersections are at x = +/- 1/sqrt(2) ~ +/- 0.707, as geometry demands.")

    _svg(os.path.join(outdir, "resultant.svg"), sylvester_matrix([1, -3, 2], [1, -5, 6]))
    print(f"\n  wrote {os.path.join(outdir, 'resultant.svg')}")


def _svg(path, M, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Sylvester matrix of (x-1)(x-2) and (x-2)(x-3): its determinant is the resultant (= 0)</text>',
    ]
    n = len(M)
    cell = 70
    ox, oy = 90, 70
    for i in range(n):
        for j in range(n):
            v = M[i][j]
            x = ox + j * cell
            y = oy + i * cell
            fill = "#161b22" if v == 0 else ("#16324d" if i < (n // 2 + n % 2) else "#3d2416")
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" fill="{fill}" '
                         f'stroke="#30363d"/>')
            parts.append(f'<text x="{x+cell/2-2:.0f}" y="{y+cell/2+4:.0f}" fill="#e6edf3" '
                         f'font-size="14" text-anchor="middle">{int(v)}</text>')
    # rows from p (blue) and q (orange)
    parts.append(f'<text x="{ox-10}" y="{oy+cell/2+4:.0f}" fill="#4dabf7" font-size="10" '
                 f'text-anchor="end">p</text>')
    parts.append(f'<text x="{ox-10}" y="{oy+(n-1)*cell+cell/2+4:.0f}" fill="#ff922b" font-size="10" '
                 f'text-anchor="end">q</text>')
    parts.append(f'<text x="{ox}" y="{oy+n*cell+24:.0f}" fill="#8b949e" font-size="11">'
                 f'det = 0  ->  the two polynomials share the root x = 2</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
