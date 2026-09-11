"""Demo: Bezier curves via de Casteljau's algorithm.

Draws quadratic and cubic Bezier curves with their control polygons, shows the de Casteljau
construction at t=0.5 (the nested linear interpolations that land on the curve), and confirms
subdivision reproduces the curve and degree elevation preserves it. Reports arc lengths.

    python examples/bezier_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bezier import (evaluate, subdivide, elevate_degree, arc_length, sample,  # noqa: E402
                    derivative, in_convex_hull)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    quad = [(0, 0), (2, 4), (4, 0)]
    cubic = [(0, 0), (1, 4), (3, -2), (4, 2)]

    print("Bezier curves: shaped by control points, evaluated by de Casteljau\n")

    print("  Quadratic Bezier, control points", quad)
    print(f"    B(0) = {evaluate(quad, 0)}  (first control point)")
    print(f"    B(0.5) = {tuple(round(v, 3) for v in evaluate(quad, 0.5))}")
    print(f"    B(1) = {evaluate(quad, 1)}  (last control point)")
    print(f"    arc length {arc_length(quad):.4f}, tangent at 0.5 = "
          f"{tuple(round(v, 2) for v in derivative(quad, 0.5))}\n")

    print("  Cubic Bezier, control points", cubic)
    print(f"    arc length {arc_length(cubic):.4f}")
    print(f"    every sampled point inside the control hull: "
          f"{all(in_convex_hull(p, cubic) for p in sample(cubic, 30))}\n")

    # subdivision
    left, right = subdivide(cubic, 0.5)
    check_sub = all(all(abs(evaluate(left, s)[d] - evaluate(cubic, s * 0.5)[d]) < 1e-12
                        for d in range(2)) for s in (0, 0.5, 1))
    print(f"  Subdivision at t=0.5 splits into two cubics that reproduce the curve: {check_sub}")
    print(f"    left control points:  {[tuple(round(v, 2) for v in p) for p in left]}")
    print(f"    right control points: {[tuple(round(v, 2) for v in p) for p in right]}\n")

    # degree elevation
    elevated = elevate_degree(quad)
    same = all(all(abs(evaluate(elevated, t)[d] - evaluate(quad, t)[d]) < 1e-12
                   for d in range(2)) for t in (0.25, 0.5, 0.75))
    print(f"  Degree elevation: the quadratic rewritten as a cubic (same shape: {same})")
    print(f"    new control points: {[tuple(round(v, 2) for v in p) for p in elevated]}\n")

    print("  de Casteljau evaluates B(t) by repeated linear interpolation of the control points:")
    print("  interpolate adjacent pairs at t, then interpolate the results, until one point remains")
    print("  -- that point is on the curve. Keeping the triangle's outer edges splits the curve in")
    print("  two. It is numerically stable and the basis of every vector-graphics renderer.")

    _svg(os.path.join(outdir, "bezier.svg"), quad, cubic)
    print(f"\n  wrote {os.path.join(outdir, 'bezier.svg')}")


def _svg(path, quad, cubic, width=760, height=430):
    allpts = quad + cubic + sample(quad, 60) + sample(cubic, 60)
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    xa, xb = min(xs) - 0.5, max(xs) + 0.5
    ya, yb = min(ys) - 0.5, max(ys) + 0.5
    px0, px1 = 45, width - 30
    py0, py1 = height - 45, 70

    def X(x):
        return px0 + (x - xa) / (xb - xa) * (px1 - px0)

    def Y(y):
        return py0 - (y - ya) / (yb - ya) * (py0 - py1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Bezier curves: control polygons (dashed) pull the curves (solid)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the curve touches only its first and last control points; the middle ones tug it, and '
        f'the de Casteljau point at t=0.5 is marked</text>',
    ]

    def draw(ctrl, curve_col, ctrl_col):
        # control polygon
        cp = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in ctrl)
        parts.append(f'<polyline points="{cp}" fill="none" stroke="{ctrl_col}" '
                     f'stroke-width="1.2" stroke-dasharray="5 3"/>')
        for p in ctrl:
            parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="4" fill="{ctrl_col}"/>')
        # the curve
        pts = sample(ctrl, 80)
        cv = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in pts)
        parts.append(f'<polyline points="{cv}" fill="none" stroke="{curve_col}" stroke-width="2.4"/>')
        # de Casteljau point at 0.5
        mid = evaluate(ctrl, 0.5)
        parts.append(f'<circle cx="{X(mid[0]):.1f}" cy="{Y(mid[1]):.1f}" r="4.5" fill="none" '
                     f'stroke="#ffd43b" stroke-width="2"/>')

    draw(quad, "#4dabf7", "#8b949e")
    draw(cubic, "#06d6a0", "#b197fc")
    parts.append(f'<text x="{px0+6}" y="{py1+2}" fill="#4dabf7" font-size="10">quadratic</text>')
    parts.append(f'<text x="{px0+6}" y="{py1+16}" fill="#06d6a0" font-size="10">cubic</text>')
    parts.append(f'<text x="{px0+6}" y="{py1+30}" fill="#ffd43b" font-size="10">B(0.5)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
