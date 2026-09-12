"""Demo: half-plane intersection -- the feasible region of linear inequalities.

Intersects a set of linear constraints into their feasible convex polygon (the region of a 2-D linear
program), reports its area and vertices, detects infeasibility, and draws the constraint lines with the
feasible region shaded.

    python examples/half_plane_intersection_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from half_plane_intersection import (intersect_half_planes, is_feasible,  # noqa: E402
                                     polygon_area)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Half-plane intersection: the feasible region of linear constraints\n")

    # a linear-program-style feasible region
    # x >= 0, y >= 0, x + y <= 6, x + 3y <= 12, 3x + y <= 15
    constraints = [
        ("x >= 0", (-1, 0, 0)),
        ("y >= 0", (0, -1, 0)),
        ("x + y <= 6", (1, 1, 6)),
        ("x + 3y <= 12", (1, 3, 12)),
        ("3x + y <= 15", (3, 1, 15)),
    ]
    hps = [hp for _, hp in constraints]
    poly = intersect_half_planes(hps)

    print("  constraints:")
    for name, _ in constraints:
        print(f"    {name}")

    print(f"\n  feasible region: a {len(poly)}-gon, area {polygon_area(poly):.3f}")
    print("  vertices (corners of the feasible polygon):")
    for x, y in poly:
        print(f"    ({x:.2f}, {y:.2f})")

    # the LP optimum of a linear objective sits at a vertex -- show maximising 3x + 2y
    best = max(poly, key=lambda p: 3 * p[0] + 2 * p[1])
    print(f"\n  maximising 3x + 2y over the region: vertex ({best[0]:.2f}, {best[1]:.2f}), "
          f"value {3*best[0] + 2*best[1]:.2f}")
    print("  (a linear objective is always optimised at a vertex -- the heart of the simplex method)")

    print(f"\n  infeasible example (x <= 0 and x >= 1): feasible? "
          f"{is_feasible([(1, 0, 0), (-1, 0, -1)])}")

    print("\n  Each inequality is a half-plane; clipping a big bounding box against them one by one")
    print("  (Sutherland-Hodgman) carves out their intersection -- a convex polygon, or empty if the")
    print("  constraints conflict. This is exactly the feasible region of a 2-D linear program.")

    _svg(os.path.join(outdir, "half_plane_intersection.svg"), constraints, poly, best)
    print(f"\n  wrote {os.path.join(outdir, 'half_plane_intersection.svg')}")


def _svg(path, constraints, poly, best, width=520, height=520):
    lo, hi = -1, 8
    ox, oy = 50, 470
    scale = 52

    def px(x):
        return ox + (x - lo) * scale

    def py(y):
        return oy - (y - lo) * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'Feasible region of 5 linear constraints (green)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="11">'
        f'each line is a constraint boundary; the optimum of a linear objective sits at a vertex</text>',
    ]

    # axes
    parts.append(f'<line x1="{px(lo)}" y1="{py(0)}" x2="{px(hi)}" y2="{py(0)}" stroke="#30363d"/>')
    parts.append(f'<line x1="{px(0)}" y1="{py(lo)}" x2="{px(0)}" y2="{py(hi)}" stroke="#30363d"/>')

    # feasible polygon (shaded)
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in poly)
    parts.append(f'<polygon points="{pts}" fill="#06d6a0" opacity="0.35" stroke="#06d6a0" '
                 f'stroke-width="2"/>')

    # constraint boundary lines
    for name, (a, b, c) in constraints:
        # draw the line a x + b y = c across the plot box
        seg = _line_seg(a, b, c, lo, hi)
        if seg:
            (x1, y1), (x2, y2) = seg
            parts.append(f'<line x1="{px(x1):.1f}" y1="{py(y1):.1f}" x2="{px(x2):.1f}" '
                         f'y2="{py(y2):.1f}" stroke="#4dabf7" stroke-width="1" opacity="0.6"/>')

    # vertices
    for x, y in poly:
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="#ffd43b"/>')
    # optimum
    parts.append(f'<circle cx="{px(best[0]):.1f}" cy="{py(best[1]):.1f}" r="7" fill="#ff6b6b"/>')
    parts.append(f'<text x="{px(best[0])+10:.0f}" y="{py(best[1]):.0f}" fill="#ff6b6b" '
                 f'font-size="11">optimum</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def _line_seg(a, b, c, lo, hi):
    """Clip the line a x + b y = c to the box [lo,hi]^2; return two endpoints or None."""
    pts = []
    if b != 0:
        for x in (lo, hi):
            y = (c - a * x) / b
            if lo - 1e-9 <= y <= hi + 1e-9:
                pts.append((x, y))
    if a != 0:
        for y in (lo, hi):
            x = (c - b * y) / a
            if lo - 1e-9 <= x <= hi + 1e-9:
                pts.append((x, y))
    # dedupe
    uniq = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in uniq):
            uniq.append(p)
    return (uniq[0], uniq[1]) if len(uniq) >= 2 else None


if __name__ == "__main__":
    main()
