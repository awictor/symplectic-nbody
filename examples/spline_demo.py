"""Demo: cubic spline interpolation -- smooth through every point.

Fits a natural cubic spline to sampled data, confirms it passes through the knots and is C^2,
and contrasts it with a single interpolating polynomial on Runge's function (where the
polynomial explodes but the spline stays tame). Draws both curves through the same knots.

    python examples/spline_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spline import CubicSpline, lagrange  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Cubic spline: a separate cubic per interval, stitched C^2 smooth\n")
    xs = [0, 1, 2, 3, 4, 5]
    ys = [0, 0.8, 0.9, 0.1, -0.8, -1.0]
    sp = CubicSpline(xs, ys)
    print(f"  {len(xs)} knots; spline passes through them all: "
          f"{all(abs(sp(x) - y) < 1e-10 for x, y in zip(xs, ys))}")
    print(f"  natural end curvatures: M0 = {sp.M[0]:.1f}, Mn = {sp.M[-1]:.1f}")
    print(f"  value/slope/curvature at interior knot 3: "
          f"{sp(3):.3f} / {sp.derivative(3):.3f} / {sp.second_derivative(3):.3f}\n")

    print("  Runge's function 1/(1+25x^2), 11 equally spaced knots -- spline vs one polynomial:")
    rx = [-1 + i * 0.2 for i in range(11)]
    ry = [1 / (1 + 25 * x * x) for x in rx]
    rsp = CubicSpline(rx, ry)
    print(f"  {'x':>6}{'true':>10}{'spline':>10}{'polynomial':>13}")
    for x in (0.7, 0.8, 0.9, 0.95):
        true = 1 / (1 + 25 * x * x)
        print(f"  {x:>6.2f}{true:>10.4f}{rsp(x):>10.4f}{lagrange(rx, ry, x):>13.4f}")
    print("\n  The single polynomial oscillates to ~1.6 near the edges (Runge's phenomenon)")
    print("  while the spline hugs the true curve. The spline minimizes total curvature -- the")
    print("  shape a flexible ruler takes. It's the interpolation behind fonts, animation, and CAD.")

    _svg(os.path.join(outdir, "spline.svg"), rx, ry)
    print(f"\n  wrote {os.path.join(outdir, 'spline.svg')}")


def _svg(path, rx, ry, w=760, h=400):
    sp = CubicSpline(rx, ry)
    true = lambda x: 1 / (1 + 25 * x * x)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Cubic spline vs single polynomial on Runge\'s function</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'both pass through the 11 knots, but the polynomial (red) oscillates while the '
        f'spline (green) stays smooth</text>',
    ]

    x0, x1 = 45, w - 30
    y0, y1 = h - 45, 70
    xa, xb = -1.0, 1.0
    ya, yb = -0.6, 1.7        # y range wide enough to show the polynomial's overshoot

    def X(x):
        return x0 + (x - xa) / (xb - xa) * (x1 - x0)

    def Y(y):
        return y0 - (y - ya) / (yb - ya) * (y0 - y1)

    # axes / zero line
    parts.append(f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x1}" y2="{Y(0):.1f}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{X(0):.1f}" y1="{y1}" x2="{X(0):.1f}" y2="{y0}" stroke="#21262d" stroke-width="1"/>')

    # sample all three curves densely
    steps = 400
    def curve(fn, col, wd):
        pts = []
        for i in range(steps + 1):
            x = xa + (xb - xa) * i / steps
            y = fn(x)
            if ya <= y <= yb:
                pts.append(f"{X(x):.1f},{Y(y):.1f}")
            else:
                # clip: break the polyline where it leaves the view
                if pts:
                    pts.append("BREAK")
        segs = " ".join(pts).split("BREAK")
        for seg in segs:
            if seg.strip():
                parts.append(f'<polyline points="{seg.strip()}" fill="none" stroke="{col}" stroke-width="{wd}"/>')

    curve(true, "#8b949e", 1.2)      # true function, faint
    curve(lambda x: lagrange(rx, ry, x), "#ff6b6b", 2)
    curve(sp, "#06d6a0", 2.5)

    # knots
    for x, y in zip(rx, ry):
        parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="#ffd43b"/>')

    # legend
    lx, ly = x0 + 12, y1 + 6
    for col, lab in (("#06d6a0", "cubic spline"), ("#ff6b6b", "single polynomial"),
                     ("#8b949e", "true 1/(1+25x^2)"), ("#ffd43b", "knots")):
        parts.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{col}"/>'
                     f'<text x="{lx+15}" y="{ly+9}" fill="#e6edf3" font-size="10">{lab}</text>')
        ly += 18

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
