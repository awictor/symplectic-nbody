"""Demo: the Henon map -- a strange attractor in two lines.

Prints the map's fixed points, area-contraction factor and Lyapunov exponent, then draws the
Henon attractor as a point cloud alongside a zoom into one arc, revealing the fractal Cantor-
set strands that make it "strange".

    python examples/henon_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from henon import (attractor_points, area_contraction, fixed_points,  # noqa: E402
                   largest_lyapunov, A_CLASSIC, B_CLASSIC)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Henon map: x'=1-a x^2+y, y'=b x  (a=%.1f, b=%.1f)\n" % (A_CLASSIC, B_CLASSIC))
    print("  Area contraction per step |det J| = |b| = %.2f (dissipative -> attractor)" % area_contraction())
    print("  Fixed points:")
    for (fx, fy) in fixed_points():
        print(f"    ({fx:+.4f}, {fy:+.4f})")
    lam = largest_lyapunov(n=30000)
    print("\n  Largest Lyapunov exponent = %.3f nat/iteration (positive -> chaos)." % lam)
    print("  Fractal (correlation) dimension ~ 1.26 -- between a curve and a filled region.\n")

    print("  The map stretches and folds the plane each step: a blob shrinks in area (by |b|)")
    print("  yet is pulled apart along the unstable direction, so it collapses onto a fractal")
    print("  of nested arcs. Zoom in and each arc is really a Cantor set of finer strands --")
    print("  the hallmark of a strange attractor, chaos with structure at every scale.")

    _svg(os.path.join(outdir, "henon.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'henon.svg')}")


def _svg(path, size=760, pad=60):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">The Henon strange attractor</text>',
        f'<text x="20" y="48" fill="#8b949e" font-size="12">'
        f'the full attractor (left) and a zoom (right) showing the fractal Cantor strands</text>',
    ]

    pts = attractor_points(20000)

    # --- left: full attractor ---
    lx0, lx1 = pad, size * 0.50
    ly0, ly1 = size - pad, pad + 40
    xmin, xmax = -1.4, 1.4
    ymin, ymax = -0.45, 0.45
    def LX(x):
        return lx0 + (x - xmin) / (xmax - xmin) * (lx1 - lx0)
    def LY(y):
        return ly0 - (y - ymin) / (ymax - ymin) * (ly0 - ly1)
    for (x, y) in pts:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            parts.append(f'<circle cx="{LX(x):.1f}" cy="{LY(y):.1f}" r="0.45" fill="#4dabf7"/>')
    parts.append(f'<rect x="{lx0:.1f}" y="{ly1:.1f}" width="{lx1-lx0:.1f}" height="{ly0-ly1:.1f}" '
                 f'fill="none" stroke="#21262d" stroke-width="1"/>')
    # zoom box on the left panel
    zx0, zx1, zy0, zy1 = 0.0, 0.7, 0.05, 0.25
    parts.append(f'<rect x="{LX(zx0):.1f}" y="{LY(zy1):.1f}" width="{LX(zx1)-LX(zx0):.1f}" '
                 f'height="{LY(zy0)-LY(zy1):.1f}" fill="none" stroke="#ffd43b" stroke-width="1"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">full attractor (yellow box = zoom region)</text>')

    # --- right: zoom into the boxed region, showing strands ---
    rx0, rx1 = size * 0.56, size - pad
    ry0, ry1 = size - pad, pad + 40
    def ZX(x):
        return rx0 + (x - zx0) / (zx1 - zx0) * (rx1 - rx0)
    def ZY(y):
        return ry0 - (y - zy0) / (zy1 - zy0) * (ry0 - ry1)
    zoom_pts = attractor_points(200000)
    drawn = 0
    for (x, y) in zoom_pts:
        if zx0 <= x <= zx1 and zy0 <= y <= zy1:
            parts.append(f'<circle cx="{ZX(x):.1f}" cy="{ZY(y):.1f}" r="0.5" fill="#ffd43b"/>')
            drawn += 1
    parts.append(f'<rect x="{rx0:.1f}" y="{ry1:.1f}" width="{rx1-rx0:.1f}" height="{ry0-ry1:.1f}" '
                 f'fill="none" stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">zoom: single arcs resolve into many parallel strands</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
