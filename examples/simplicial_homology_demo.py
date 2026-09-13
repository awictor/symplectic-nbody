"""Demo: computing the homology of standard spaces from their triangulations.

Runs the boundary-matrix + Smith-Normal-Form pipeline on a circle, sphere, torus, and projective plane,
printing the Betti numbers (hole counts), torsion, Euler characteristic, and the Euler-Poincare check.
Draws a table of Betti numbers per dimension.

    python examples/simplicial_homology_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simplicial_homology import (  # noqa: E402
    homology, euler_characteristic, euler_from_betti, verify_boundary_squared_zero,
    circle, disk, sphere, figure_eight, torus, projective_plane,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Simplicial homology: counting holes via boundary matrices + Smith Normal Form\n")

    spaces = [
        ("circle S^1", circle(), "1 loop"),
        ("disk D^2", disk(), "contractible"),
        ("sphere S^2", sphere(), "1 void, no loops"),
        ("figure-eight", figure_eight(), "2 loops"),
        ("torus T^2", torus(), "2 loops, 1 void"),
        ("proj plane RP^2", projective_plane(), "Z/2 torsion (non-orientable)"),
    ]

    print(f"    {'space':<16}{'V':>3}{'E':>4}{'F':>4}   {'Betti (b0,b1,b2)':<18}"
          f"{'torsion':<10}{'chi':>4}")
    rows = []
    for name, cx, note in spaces:
        b, t = homology(cx)
        V = len(cx.get(0, []))
        E = len(cx.get(1, []))
        F = len(cx.get(2, []))
        betti = tuple(b[k] for k in sorted(b))
        tor = [f"H{k}:Z/{v}" for k in t for v in t[k]]
        chi = euler_characteristic(cx)
        rows.append((name, betti, note))
        assert chi == euler_from_betti(b), "Euler-Poincare mismatch"
        assert verify_boundary_squared_zero(cx), "d^2 != 0"
        print(f"    {name:<16}{V:>3}{E:>4}{F:>4}   {str(betti):<18}"
              f"{(','.join(tor) or '-'):<10}{chi:>4}")

    print(f"\n  b_0 = connected components, b_1 = independent loops, b_2 = enclosed voids.")
    print(f"  The torus has b_1 = 2 (the two independent loops around it); RP^2 has a Z/2")
    print(f"  torsion class -- a loop that is not null-homotopic but whose DOUBLE bounds.")
    print(f"  Every space passes d^2 = 0 and the Euler-Poincare identity chi = sum (-1)^k b_k.")

    _svg(os.path.join(outdir, "simplicial_homology.svg"), rows)
    print(f"\n  wrote {os.path.join(outdir, 'simplicial_homology.svg')}")


def _svg(path, rows, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Betti numbers by dimension: components (b0), loops (b1), voids (b2)</text>',
    ]
    ox, oy = 170, 60
    row_h = 44
    colw = 70
    dims = 3
    colors = ["#4dabf7", "#06d6a0", "#ffd43b"]
    labels = ["b0 (components)", "b1 (loops)", "b2 (voids)"]
    for d in range(dims):
        parts.append(f'<text x="{ox + d*colw + colw/2:.0f}" y="{oy-10}" fill="{colors[d]}" '
                     f'font-size="10" text-anchor="middle">b{d}</text>')
    for i, (name, betti, note) in enumerate(rows):
        y = oy + i * row_h
        parts.append(f'<text x="{ox-12}" y="{y+16:.0f}" fill="#e6edf3" font-size="11" '
                     f'text-anchor="end">{name}</text>')
        for d in range(dims):
            v = betti[d] if d < len(betti) else 0
            x = ox + d * colw
            # bubble sized by Betti value
            r = 6 + 7 * v
            col = colors[d] if v > 0 else "#30363d"
            parts.append(f'<circle cx="{x+colw/2:.0f}" cy="{y+12:.0f}" r="{min(r,20)}" fill="{col}"/>')
            parts.append(f'<text x="{x+colw/2:.0f}" y="{y+16:.0f}" fill="#0d1117" font-size="11" '
                         f'text-anchor="middle">{v}</text>')
        parts.append(f'<text x="{ox + dims*colw + 15:.0f}" y="{y+16:.0f}" fill="#8b949e" '
                     f'font-size="9">{note}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
