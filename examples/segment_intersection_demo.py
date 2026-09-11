"""Demo: line-segment intersection and the simple-polygon test.

Classifies pairs of segments (crossing, touching, collinear-overlap, parallel, disjoint) with the
orientation predicate, finds crossing points, and tests whether polygons are simple (no
non-adjacent edges cross). Draws a simple polygon beside a self-intersecting one.

    python examples/segment_intersection_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segment_intersection import (segments_intersect, intersection_point,  # noqa: E402
                                  is_simple_polygon, count_intersections, orientation)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Line-segment intersection via the orientation predicate\n")

    cases = [
        ("proper X-crossing", (0, 0), (4, 4), (0, 4), (4, 0)),
        ("T-touch (endpoint on edge)", (0, 0), (4, 0), (2, 0), (2, 3)),
        ("shared endpoint", (0, 0), (2, 2), (2, 2), (4, 0)),
        ("collinear overlap", (0, 0), (4, 0), (2, 0), (6, 0)),
        ("parallel, apart", (0, 0), (4, 0), (0, 1), (4, 1)),
        ("disjoint", (0, 0), (1, 1), (3, 0), (4, 1)),
    ]
    print(f"    {'case':>28}  {'intersect?':>10}  crossing point")
    for name, a, b, c, d in cases:
        hit = segments_intersect(a, b, c, d)
        pt = intersection_point(a, b, c, d)
        pts = f"({pt[0]:.1f}, {pt[1]:.1f})" if pt else "-- (none / not a single point)"
        print(f"    {name:>28}  {str(hit):>10}  {pts}")

    print("\n  Simple-polygon test (no non-adjacent edges cross -- the precondition for area/PIP):")
    polys = [
        ("square", [(0, 0), (2, 0), (2, 2), (0, 2)]),
        ("convex pentagon", [(0, 0), (4, 0), (5, 3), (2, 5), (-1, 3)]),
        ("arrow (non-convex)", [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]),
        ("figure-eight (self-crossing)", [(0, 0), (2, 2), (2, 0), (0, 2)]),
    ]
    for name, verts in polys:
        print(f"    {name:>30}: simple = {is_simple_polygon(verts)}")

    # a grid of segments, count crossings
    horiz = [((0, y), (6, y)) for y in range(4)]
    vert = [((x, 0), (x, 4)) for x in range(6)]
    count, _ = count_intersections(horiz + vert)
    print(f"\n  A {len(horiz)}x{len(vert)} grid of horizontal/vertical segments has {count} "
          f"crossings ({len(horiz)}*{len(vert)}).")

    print("\n  orient(a,b,c) = sign of the cross product tells which side of line ab point c is on.")
    print("  Two segments properly cross iff each straddles the other's line (opposite orientations")
    print("  at its endpoints); zero orientations mean collinear/touching, settled by a bounding-box")
    print("  check. No slopes, so vertical segments are no trouble -- the robust geometric predicate.")

    _svg(os.path.join(outdir, "segment_intersection.svg"), polys)
    print(f"\n  wrote {os.path.join(outdir, 'segment_intersection.svg')}")


def _svg(path, polys, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Simple vs self-intersecting polygons</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'green = simple (no non-adjacent edges cross); red = self-intersecting</text>',
    ]
    # draw four polygons in a 2x2 grid of panels
    panels = [(0, 0), (1, 0), (0, 1), (1, 1)]
    pw, ph = width / 2, (height - 70) / 2
    for (verts_name, (name, verts)), (gx, gy) in zip(enumerate(polys), panels):
        simple = is_simple_polygon(verts)
        col = "#06d6a0" if simple else "#ff6b6b"
        ox = 40 + gx * pw
        oy = 80 + gy * ph
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        xa, xb = min(xs), max(xs)
        ya, yb = min(ys), max(ys)
        sx = (pw - 90) / (xb - xa) if xb > xa else 1
        sy = (ph - 50) / (yb - ya) if yb > ya else 1
        sc = min(sx, sy)

        def PX(x):
            return ox + 30 + (x - xa) * sc

        def PY(y):
            return oy + ph - 40 - (y - ya) * sc

        n = len(verts)
        pts = " ".join(f"{PX(verts[i][0]):.1f},{PY(verts[i][1]):.1f}" for i in range(n))
        parts.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.12" '
                     f'stroke="{col}" stroke-width="2"/>')
        for v in verts:
            parts.append(f'<circle cx="{PX(v[0]):.1f}" cy="{PY(v[1]):.1f}" r="3" fill="{col}"/>')
        tag = "simple" if simple else "self-intersecting"
        parts.append(f'<text x="{ox+30:.1f}" y="{oy+14:.1f}" fill="#e6edf3" font-size="11">'
                     f'{name} ({tag})</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
