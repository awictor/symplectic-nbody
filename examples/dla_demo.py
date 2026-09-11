"""Demo: diffusion-limited aggregation -- fractal growth from random walkers.

Grows a DLA cluster, prints its size / radius of gyration / fractal dimension, and draws the
feathery cluster next to the mass-radius scaling N(r) ~ r^D that measures its dimension.

    python examples/dla_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dla import (grow, radius_of_gyration, fractal_dimension,  # noqa: E402
                 center_of_mass, bounds)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Diffusion-limited aggregation: random walkers stick to a growing seed\n")
    print(f"  {'particles':>10}{'R_gyration':>14}{'fractal D':>12}")
    for n in (200, 600, 1200):
        c = grow(n, seed=3)
        D, _ = fractal_dimension(c)
        print(f"  {n:>10}{radius_of_gyration(c):>14.2f}{D:>12.3f}")

    print("\n  Mass grows as N(r) ~ r^D with D ~ 1.7 in the plane, not 2: the branches")
    print("  leave most of the plane empty. Outer tips screen the interior fjords -- a")
    print("  wanderer brushes a tip long before it diffuses inside -- so the tips grow")
    print("  faster and the cluster stays sparse and self-similar. Mineral dendrites,")
    print("  electrodeposits, viscous fingers, lightning, and soot all grow this way.")

    _svg(os.path.join(outdir, "dla.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'dla.svg')}")


def _svg(path, size=760):
    cluster = grow(1400, seed=3)
    D, pts = fractal_dimension(cluster)
    cx, cy = center_of_mass(cluster)
    xmin, ymin, xmax, ymax = bounds(cluster)
    span = max(xmax - xmin, ymax - ymin) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size//2+70}" '
        f'viewBox="0 0 {size} {size//2+70}" font-family="monospace">',
        f'<rect width="{size}" height="{size//2+70}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Diffusion-limited aggregation: a fractal from random walkers</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'the cluster (left) and its mass-radius scaling N(r) ~ r^D (right); '
        f'D = {D:.2f}, not 2</text>',
    ]

    # left: the cluster, coloured by distance from centre (tip vs core)
    panel = size // 2 - 40
    px0, py0 = 30, 60
    sc = panel / span
    r_out = max(math.hypot(a - cx, b - cy) for a, b in cluster)
    for (x, y) in cluster:
        sx = px0 + (x - xmin) * sc
        sy = py0 + (y - ymin) * sc
        frac = math.hypot(x - cx, y - cy) / (r_out or 1)
        # core purple -> tips orange
        col = "#8338ec" if frac < 0.33 else ("#b197fc" if frac < 0.66 else "#ff922b")
        parts.append(f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{max(sc,1.4):.1f}" '
                     f'height="{max(sc,1.4):.1f}" fill="{col}"/>')

    # right: log-log mass-radius with the fitted slope D
    gx0, gx1 = size // 2 + 40, size - 40
    gy0, gy1 = size // 2 + 20, 70
    lr = [math.log(r) for r, _ in pts]
    ln = [math.log(n) for _, n in pts]
    lrmin, lrmax = min(lr), max(lr)
    lnmin, lnmax = min(ln), max(ln)

    def GX(v):
        return gx0 + (v - lrmin) / (lrmax - lrmin) * (gx1 - gx0)

    def GY(v):
        return gy0 - (v - lnmin) / (lnmax - lnmin) * (gy0 - gy1)

    parts.append(f'<line x1="{gx0}" y1="{gy0}" x2="{gx1}" y2="{gy0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{gx0}" y1="{gy0}" x2="{gx0}" y2="{gy1}" stroke="#8b949e" stroke-width="1.2"/>')
    # reference slope-2 line (solid disc) from the first point
    r0, n0 = lr[0], ln[0]
    x2a, x2b = lrmin, lrmax
    parts.append(f'<line x1="{GX(x2a):.1f}" y1="{GY(n0 + 2*(x2a-r0)):.1f}" '
                 f'x2="{GX(x2b):.1f}" y2="{GY(n0 + 2*(x2b-r0)):.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{GX(x2b):.1f}" y="{GY(n0 + 2*(x2b-r0))-4:.1f}" fill="#8b949e" '
                 f'font-size="9" text-anchor="end">slope 2 (disc)</text>')
    # data points + fitted slope-D line
    line = " ".join(f"{GX(r):.1f},{GY(n):.1f}" for r, n in zip(lr, ln))
    parts.append(f'<polyline points="{line}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for r, n in zip(lr, ln):
        parts.append(f'<circle cx="{GX(r):.1f}" cy="{GY(n):.1f}" r="2.5" fill="#4dabf7"/>')
    parts.append(f'<text x="{(gx0+gx1)/2:.1f}" y="{gy0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">log r</text>')
    parts.append(f'<text x="{gx0-8:.1f}" y="{gy1+2:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">log N</text>')
    parts.append(f'<text x="{GX((lrmin+lrmax)/2):.1f}" y="{GY((lnmin+lnmax)/2)-8:.1f}" '
                 f'fill="#06d6a0" font-size="12">D = {D:.2f}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
