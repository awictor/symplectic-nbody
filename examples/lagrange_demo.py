"""Demo: Lagrange points and zero-velocity (Hill) curves of the CR3BP.

Prints the five Lagrange-point locations for the Earth-Moon system and renders
an SVG of the effective-potential landscape: the primaries, the five L-points,
and the zero-velocity curves (contours of the Jacobi constant) that bound where
a particle of a given energy can go.

    python examples/lagrange_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cr3bp import CR3BP  # noqa: E402

_PALETTE = {"L1": "#e63946", "L2": "#f4a261", "L3": "#e9c46a",
            "L4": "#2a9d8f", "L5": "#3a86ff"}


def render_hill(c: CR3BP, path: str, C_levels, size=760, pad=40,
                extent=1.6, grid=240):
    """Marching-squares contours of C(x,y) = 2*Omega(x,y) at each level."""
    xs = [-extent + 2 * extent * i / (grid - 1) for i in range(grid)]
    ys = xs
    # field value = 2*Omega (the max attainable Jacobi C at zero velocity)
    field = [[2.0 * c.effective_potential(x, y) for x in xs] for y in ys]

    def sx(x):
        return pad + (x + extent) / (2 * extent) * (size - 2 * pad)

    def sy(y):
        return size - (pad + (y + extent) / (2 * extent) * (size - 2 * pad))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]

    # marching squares: draw a segment wherever an edge crosses the level
    for level in C_levels:
        segs = []
        for j in range(grid - 1):
            for i in range(grid - 1):
                cell = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                vals = [field[jj][ii] for ii, jj in cell]
                for k in range(4):
                    a, b = k, (k + 1) % 4
                    va, vb = vals[a], vals[b]
                    if (va - level) * (vb - level) < 0:
                        t = (level - va) / (vb - va)
                        ia, ja = cell[a]; ib, jb = cell[b]
                        px = xs[ia] + t * (xs[ib] - xs[ia])
                        py = ys[ja] + t * (ys[jb] - ys[ja])
                        segs.append((px, py))
        # draw crossings as tiny dots (cheap, dependency-free contour)
        pts = " ".join(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="0.7" '
                       f'fill="#30506e"/>' for px, py in segs)
        parts.append(pts)

    # primaries
    parts.append(f'<circle cx="{sx(-c.mu):.1f}" cy="{sy(0):.1f}" r="7" fill="#cbd5e1"/>')
    parts.append(f'<circle cx="{sx(1-c.mu):.1f}" cy="{sy(0):.1f}" r="4" fill="#94a3b8"/>')

    # Lagrange points
    for name, (x, y) in c.lagrange_points().items():
        col = _PALETTE[name]
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{sx(x)+8:.1f}" y="{sy(y)-6:.1f}" fill="{col}" '
                     f'font-size="13">{name}</text>')

    parts.append(f'<text x="{pad}" y="28" fill="#e6edf3" font-size="18">'
                 f'CR3BP Lagrange points &amp; zero-velocity curves</text>')
    parts.append(f'<text x="{pad}" y="48" fill="#8b949e" font-size="12">'
                 f'Earth-Moon mu={c.mu}; dots trace contours of the Jacobi constant</text>')
    parts.append("</svg>")

    svg = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return svg


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    c = CR3BP(0.01215)  # Earth-Moon

    print("Earth-Moon CR3BP (mu = 0.01215), nondimensional units\n")
    print(f"{'point':<6}{'x':>12}{'y':>12}{'Jacobi C':>14}")
    print("-" * 44)
    for name, (x, y) in c.lagrange_points().items():
        C = c.jacobi_constant(x, y, 0.0, 0.0)
        print(f"{name:<6}{x:>12.6f}{y:>12.6f}{C:>14.6f}")

    # contour levels near the collinear-point energies
    Cs = [c.jacobi_constant(*c.lagrange_points()[p], 0.0, 0.0)
          for p in ("L1", "L2", "L3")]
    path = os.path.join(outdir, "lagrange.svg")
    render_hill(c, path, Cs)
    print(f"\nwrote {path}")
    print("L4/L5 (equilateral points) are stable for the Earth-Moon mass ratio;")
    print("that's why Trojan asteroids cluster at the Sun-Jupiter L4/L5.")


if __name__ == "__main__":
    main()
