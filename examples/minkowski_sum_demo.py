"""Minkowski sum demo: A (+) B of two convex polygons, and collision via the Minkowski difference (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import minkowski_sum as MS
from convex_hull import polygon_area


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
RED = "#ff6b6b"
PURPLE = "#b197fc"


def main(outdir=None):
    A = [(0, 0), (3, 0), (4, 2), (2, 3), (0, 2)]     # convex pentagon
    B = [(0, 0), (1.5, 0), (1.5, 1.5), (0, 1.5)]     # small square
    s = MS.minkowski_sum_convex(A, B)

    lines = []
    lines.append("Minkowski sum: A (+) B = { a + b }")
    lines.append("=" * 50)
    lines.append(f"A: convex {len(A)}-gon, area {polygon_area(A):.3f}")
    lines.append(f"B: convex {len(B)}-gon, area {polygon_area(B):.3f}")
    lines.append(f"A (+) B: {len(s)}-gon, area {MS.area(s):.3f}")
    lines.append(f"  (area >= area(A)+area(B) = {polygon_area(A)+polygon_area(B):.3f}; "
                 f"the difference is the mixed 'sweep' region)")
    lines.append("")
    lines.append("edge-merge O(n+m) matches the brute-force all-pairs-plus-hull sum:")
    b = MS.minkowski_sum_brute(A, B)
    lines.append(f"  edge-merge area {MS.area(s):.4f}  brute area {MS.area(b):.4f}  "
                 f"match: {abs(MS.area(s)-MS.area(b))<1e-9}")
    lines.append("")
    lines.append("Collision detection via the Minkowski difference (two shapes overlap iff")
    lines.append("the origin lies in A (+) (-B)):")
    P = [(0, 0), (2, 0), (2, 2), (0, 2)]
    for label, Q in [("overlapping", [(1, 1), (3, 1), (3, 3), (1, 3)]),
                     ("touching   ", [(2, 0), (4, 0), (4, 2), (2, 2)]),
                     ("disjoint   ", [(5, 5), (7, 5), (7, 7), (5, 7)])]:
        lines.append(f"  {label}: intersect = {MS.convex_intersect(P, Q)}")
    lines.append("")
    lines.append("This is the geometry behind robot motion planning (grow obstacles by the robot")
    lines.append("shape, shrink the robot to a point) and GJK collision detection.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 700, 380
        # left: A, B, and A(+)B; right: Minkowski difference collision picture
        lx, ly, sc = 40, 60, 26

        def sxl(x):
            return lx + (x + 1) * sc

        def syl(y):
            return ly + (10 - y) * sc

        s_svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s_svg.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s_svg.append(f'<text x="{lx}" y="30" fill="{TEXT}" font-size="15">'
                     f'A (+) B: the sum sweeps B around A</text>')

        def poly(pts, col, fill_op, sxf, syf, wid=2):
            d = " ".join(f"{sxf(x):.1f},{syf(y):.1f}" for x, y in pts)
            return (f'<polygon points="{d}" fill="{col}" fill-opacity="{fill_op}" '
                    f'stroke="{col}" stroke-width="{wid}"/>')

        # the sum (big, faint), then A and B on top
        s_svg.append(poly(s, PURPLE, 0.18, sxl, syl, 2.5))
        s_svg.append(poly(A, BLUE, 0.25, sxl, syl))
        s_svg.append(poly(B, GREEN, 0.35, sxl, syl))
        s_svg.append(f'<text x="{lx}" y="{H-52}" fill="{BLUE}" font-size="11">A (pentagon)</text>')
        s_svg.append(f'<text x="{lx}" y="{H-38}" fill="{GREEN}" font-size="11">B (square)</text>')
        s_svg.append(f'<text x="{lx}" y="{H-24}" fill="{PURPLE}" font-size="11">A (+) B (the sum)</text>')

        # right panel: Minkowski difference collision
        rx, ry = 400, 60

        def sxr(x):
            return rx + (x + 4) * 18

        def syr(y):
            return ry + (10 - y) * 18
        Pc = [(0, 0), (2, 0), (2, 2), (0, 2)]
        Qc = [(1, 1), (3, 1), (3, 3), (1, 3)]
        diff = MS.minkowski_difference(Pc, Qc)
        s_svg.append(f'<text x="{rx}" y="30" fill="{TEXT}" font-size="14">'
                     f'collision: origin in A(+)(-B)?</text>')
        # axes
        s_svg.append(f'<line x1="{sxr(-4):.1f}" y1="{syr(0):.1f}" x2="{sxr(6):.1f}" y2="{syr(0):.1f}" '
                     f'stroke="#30363d"/>')
        s_svg.append(f'<line x1="{sxr(0):.1f}" y1="{syr(-4):.1f}" x2="{sxr(0):.1f}" y2="{syr(6):.1f}" '
                     f'stroke="#30363d"/>')
        s_svg.append(poly(diff, YELLOW, 0.2, sxr, syr, 2))
        # origin marker: red if inside (collision)
        inside = MS.convex_intersect(Pc, Qc)
        col = RED if inside else GREEN
        s_svg.append(f'<circle cx="{sxr(0):.1f}" cy="{syr(0):.1f}" r="5" fill="{col}"/>')
        s_svg.append(f'<text x="{sxr(0)+8:.1f}" y="{syr(0):.1f}" fill="{col}" font-size="11">'
                     f'origin {"inside -> collision" if inside else "outside"}</text>')
        s_svg.append(f'<text x="{rx}" y="{H-24}" fill="{GRAY}" font-size="10">'
                     f'The yellow region is A(+)(-B); the origin falling inside it means A and B overlap.</text>')
        s_svg.append("</svg>")
        with open(os.path.join(outdir, "minkowski_sum.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s_svg))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
