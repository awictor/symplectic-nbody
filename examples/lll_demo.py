"""Demo: LLL lattice reduction -- short, near-orthogonal bases and integer relations.

Reduces a badly-skewed 2D lattice basis and draws the original vs reduced basis vectors over the
lattice points, shows a 3D reduction, and recovers an integer relation among reals.

    python examples/lll_demo.py [output_dir]
"""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lll import lll_reduce, shortest_vector, integer_relation, vector_norm_sq, is_reduced  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("LLL lattice reduction: from a skewed basis to a short, near-orthogonal one\n")

    skew = [[201, 37], [98, 18]]
    red = lll_reduce(skew)
    print("  2D lattice (same points, different basis):")
    print(f"    original basis  {skew}")
    print(f"      vector norms^2: {[vector_norm_sq(v) for v in skew]}")
    print(f"    reduced basis   {red}")
    print(f"      vector norms^2: {[vector_norm_sq(v) for v in red]}")
    print(f"    reduced? {is_reduced(red)}   shortest vector: {shortest_vector(skew)}\n")

    basis3 = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    red3 = lll_reduce(basis3)
    print("  3D lattice:")
    print(f"    original {basis3}, total norm^2 {sum(vector_norm_sq(v) for v in basis3)}")
    print(f"    reduced  {red3}, total norm^2 {sum(vector_norm_sq(v) for v in red3)}")
    print(f"    reduced? {is_reduced(red3)}\n")

    print("  Integer relation detection (find small a_i with sum a_i x_i = 0):")
    reals = [6.0, 3.0, 4.0]
    rel = integer_relation(reals, scale=10 ** 8)
    print(f"    reals {reals}")
    print(f"    found relation {rel}:  {rel[0]}*6 + {rel[1]}*3 + {rel[2]}*4 = "
          f"{sum(a*r for a, r in zip(rel, reals)):.1f}\n")

    print("  A lattice has many bases -- any two related by an integer matrix of determinant +/-1")
    print("  span the same points. LLL swaps and size-reduces until the basis is short and nearly")
    print("  orthogonal, in polynomial time, all arithmetic exact over the rationals.")

    _svg(os.path.join(outdir, "lll.svg"), skew, red)
    print(f"\n  wrote {os.path.join(outdir, 'lll.svg')}")


def _svg(path, orig, red, width=760, height=420):
    cx, cy = width // 2, height // 2 + 10
    scale = 2.4  # px per lattice unit (skew vectors ~200 long -> keep small)

    def sx(x):
        return cx + x * scale

    def sy(y):
        return cy - y * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'LLL reduction of a 2D lattice basis</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'grey dots = lattice points; red = skewed original basis; green = reduced basis</text>',
    ]
    # lattice points from integer combos of the reduced (short) basis
    b0, b1 = red
    pts = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i * b0[0] + j * b1[0]
            y = i * b0[1] + j * b1[1]
            px, py = sx(x), sy(y)
            if 55 <= px <= width - 15 and 60 <= py <= height - 15:
                pts.append((px, py))
    for px, py in pts:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2" fill="#8b949e" opacity="0.6"/>')

    # axes
    parts.append(f'<line x1="55" y1="{cy}" x2="{width-15}" y2="{cy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{cx}" y1="60" x2="{cx}" y2="{height-15}" stroke="#30363d" stroke-width="1"/>')

    def arrow(v, colour, label):
        x2, y2 = sx(v[0]), sy(v[1])
        out = [f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" '
               f'stroke-width="2.5"/>',
               f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="3.5" fill="{colour}"/>',
               f'<text x="{x2:.0f}" y="{y2-6:.0f}" fill="{colour}" font-size="11" '
               f'text-anchor="middle">{label}</text>']
        return out

    for v in orig:
        parts += arrow(v, "#ff6b6b", f"({v[0]},{v[1]})")
    for v in red:
        parts += arrow(v, "#06d6a0", f"({v[0]},{v[1]})")

    parts.append(f'<text x="20" y="{height-12}" fill="#8b949e" font-size="11">'
                 f'same lattice, but the reduced basis vectors are short and nearly perpendicular</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
