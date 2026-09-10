"""Demo: a Poincare surface-of-section of the CR3BP.

Many orbits at the SAME Jacobi energy are integrated and their y=0 crossings
overlaid on the (x, vx) plane. Regular orbits draw smooth closed curves (KAM
tori); chaotic orbits scatter. One picture, order and chaos at one energy.

    python examples/poincare_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cr3bp import CR3BP  # noqa: E402
from poincare import section, vy_from_jacobi  # noqa: E402

_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
            "#8338ec", "#3a86ff", "#ff006e", "#06d6a0", "#ffbe0b",
            "#b5179e", "#4cc9f0"]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    mu, C = 0.1, 3.9
    model = CR3BP(mu)
    print(f"Poincare section of the CR3BP (mu={mu}, Jacobi C={C}, surface y=0)\n")

    # a spread of seeds across the allowed band on the section
    seeds = []
    x = -0.55
    while x <= -0.18:
        seeds.append((round(x, 3), 0.0))
        x += 0.025
    seeds += [(-0.4, 0.4), (-0.4, 0.8), (-0.3, 0.5), (-0.45, 0.25)]

    orbits = []
    for (x0, vx0) in seeds:
        if math.isnan(vy_from_jacobi(model, x0, vx0, C)):
            continue
        pts = section(model, x0, vx0, C, n_crossings=140, dt=0.002)
        if len(pts) >= 10:
            xspread = max(p[0] for p in pts) - min(p[0] for p in pts)
            orbits.append((pts, xspread))

    regular = sum(1 for _, s in orbits if s < 0.05)
    print(f"integrated {len(orbits)} orbits; {regular} trace tight closed curves "
          f"(KAM tori), the rest fill chaotic regions.")

    # render the section
    all_x = [p[0] for pts, _ in orbits for p in pts]
    all_v = [p[1] for pts, _ in orbits for p in pts]
    xmin, xmax = min(all_x), max(all_x)
    vmin, vmax = min(all_v), max(all_v)
    size, pad = 760, 56

    def sx(x):
        return pad + (x - xmin) / ((xmax - xmin) or 1) * (size - 2 * pad)

    def sy(v):
        return size - (pad + (v - vmin) / ((vmax - vmin) or 1) * (size - 2 * pad))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    for k, (pts, _s) in enumerate(orbits):
        col = _PALETTE[k % len(_PALETTE)]
        dots = "".join(f'<circle cx="{sx(px):.1f}" cy="{sy(pv):.1f}" r="1.3" '
                       f'fill="{col}"/>' for px, pv in pts)
        parts.append(dots)
    parts.append(f'<text x="{pad}" y="30" fill="#e6edf3" font-size="18">'
                 f'Poincare section: CR3BP at fixed energy</text>')
    parts.append(f'<text x="{pad}" y="50" fill="#8b949e" font-size="12">'
                 f'mu={mu}, Jacobi C={C}, surface y=0; axes (x, vx). '
                 f'closed loops = tori, scatter = chaos</text>')
    parts.append("</svg>")
    path = os.path.join(outdir, "poincare_section.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
