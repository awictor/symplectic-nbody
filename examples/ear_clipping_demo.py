"""Demo: ear-clipping triangulation of concave polygons.

Triangulates a star and an L-shape, confirms the triangle count (n-2) and exact area conservation,
and draws the polygons with their triangulation.

    python examples/ear_clipping_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ear_clipping import triangulate, polygon_area, triangle_area  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Ear-clipping triangulation: any simple polygon into n-2 triangles\n")

    shapes = {}

    # a 5-point star (concave)
    star = []
    for i in range(10):
        r = 2.0 if i % 2 == 0 else 0.8
        a = math.pi / 2 + i * math.pi / 5
        star.append((r * math.cos(a), r * math.sin(a)))
    shapes["star"] = star

    # an L-shape (concave)
    shapes["L-shape"] = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]

    # an arrow (concave)
    shapes["arrow"] = [(0, 1), (2, 1), (2, 2), (4, 0), (2, -2), (2, -1), (0, -1)]

    for name, poly in shapes.items():
        tris = triangulate(poly)
        area_tri = sum(triangle_area(*t) for t in tris)
        print(f"  {name}: {len(poly)} vertices -> {len(tris)} triangles (n-2 = {len(poly)-2})")
        print(f"    polygon area {polygon_area(poly):.4f}, triangle sum {area_tri:.4f}, "
              f"match {abs(area_tri - polygon_area(poly)) < 1e-9}")

    print("\n  The two-ears theorem guarantees every simple polygon with >3 vertices has an EAR --")
    print("  a convex vertex whose diagonal stays inside and whose triangle holds no other vertex.")
    print("  Clip the ear, repeat on the smaller polygon; n-2 triangles later, done. O(n^2).")

    _svg(os.path.join(outdir, "ear_clipping.svg"), shapes)
    print(f"\n  wrote {os.path.join(outdir, 'ear_clipping.svg')}")


def _svg(path, shapes, width=760, height=430):
    colors = ["#4dabf7", "#ffd43b", "#ff6b6b", "#06d6a0", "#b197fc", "#ff922b",
              "#e6edf3", "#8b949e"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Ear-clipping triangulation of concave polygons</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each polygon split into n-2 triangles (alternating fills); the outline is drawn in white</text>',
    ]

    names = list(shapes.keys())
    panel_w = width / len(names)
    for pi, name in enumerate(names):
        poly = shapes[name]
        tris = triangulate(poly)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        span = max(maxx - minx, maxy - miny) or 1
        scale = (panel_w - 40) / span
        cx = pi * panel_w + panel_w / 2
        cy = 240
        px0 = (minx + maxx) / 2
        py0 = (miny + maxy) / 2

        def tx(x):
            return cx + (x - px0) * scale

        def ty(y):
            return cy - (y - py0) * scale

        for ti, tr in enumerate(tris):
            pts = " ".join(f"{tx(v[0]):.1f},{ty(v[1]):.1f}" for v in tr)
            col = colors[ti % len(colors)]
            parts.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.35" '
                         f'stroke="{col}" stroke-width="1"/>')
        # outline
        outline = " ".join(f"{tx(p[0]):.1f},{ty(p[1]):.1f}" for p in poly)
        parts.append(f'<polygon points="{outline}" fill="none" stroke="#e6edf3" stroke-width="2"/>')
        parts.append(f'<text x="{cx:.0f}" y="{cy+150:.0f}" fill="#8b949e" font-size="12" '
                     f'text-anchor="middle">{name}: {len(tris)} triangles</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
