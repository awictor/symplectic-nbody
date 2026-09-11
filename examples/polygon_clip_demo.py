"""Demo: Sutherland-Hodgman polygon clipping.

Clips a subject polygon against a rectangular window and against a triangular and a diamond window,
reporting the retained area and confirming the clipped polygon never grows. Draws the subject, the
clip window, and the resulting intersection.

    python examples/polygon_clip_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from polygon_clip import clip_polygon, clip_to_rectangle, polygon_area  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # a concave star-ish subject
    subject = [(1, 1), (9, 2), (7, 5), (9, 9), (4, 7), (1, 9), (3, 5)]
    print("Sutherland-Hodgman polygon clipping: subject INTERSECT convex window\n")
    print(f"  subject polygon: {len(subject)} vertices, area {polygon_area(subject):.2f}\n")

    window = (2, 2, 8, 8)
    clipped = clip_to_rectangle(subject, *window)
    print(f"  clip to rectangle {window}: {len(clipped)} vertices, "
          f"area {polygon_area(clipped):.2f}")
    print(f"    retained {100 * polygon_area(clipped) / polygon_area(subject):.0f}% of the "
          f"subject's area; clipped area <= original: "
          f"{polygon_area(clipped) <= polygon_area(subject)}\n")

    # a few more windows
    print("  Clipping the same subject to different convex windows:")
    windows = [
        ("rectangle [2,8]^2", [(2, 2), (8, 2), (8, 8), (2, 8)]),
        ("triangle", [(0, 0), (10, 0), (5, 10)]),
        ("diamond", [(5, 0), (10, 5), (5, 10), (0, 5)]),
        ("tiny centre box", [(4, 4), (6, 4), (6, 6), (4, 6)]),
    ]
    for name, clip in windows:
        r = clip_polygon(subject, clip)
        print(f"    {name:>18}: clipped area {polygon_area(r):6.2f} "
              f"({len(r)} vertices)")

    # edge cases
    print("\n  Edge cases:")
    inside = [(3, 3), (5, 3), (5, 5), (3, 5)]
    print(f"    fully inside the window -> unchanged (area {polygon_area(clip_to_rectangle(inside, 0, 0, 10, 10)):.1f})")
    outside = [(20, 20), (22, 20), (21, 22)]
    print(f"    fully outside -> empty ({clip_to_rectangle(outside, 0, 0, 10, 10)})")

    print("\n  The algorithm clips the subject against each clip edge in turn, feeding the output")
    print("  of one edge into the next. Per edge it keeps vertices on the inside and adds the")
    print("  crossing point wherever an edge exits or enters -- O(n*k) for an n-gon and k-edge")
    print("  convex window. This is how a renderer discards geometry outside the viewport.")

    _svg(os.path.join(outdir, "polygon_clip.svg"), subject,
         [(2, 2), (8, 2), (8, 8), (2, 8)], clipped)
    print(f"\n  wrote {os.path.join(outdir, 'polygon_clip.svg')}")


def _svg(path, subject, window, clipped, width=760, height=430):
    allpts = subject + window + clipped
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    xa, xb = min(xs) - 1, max(xs) + 1
    ya, yb = min(ys) - 1, max(ys) + 1
    px0, px1 = 45, width - 30
    py0, py1 = height - 40, 70

    def X(x):
        return px0 + (x - xa) / (xb - xa) * (px1 - px0)

    def Y(y):
        return py0 - (y - ya) / (yb - ya) * (py0 - py1)

    def poly_pts(poly):
        return " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in poly)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Polygon clipping: subject (grey) clipped to a window (yellow) -> result (green)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'Sutherland-Hodgman keeps the intersection of the subject polygon and the convex '
        f'window</text>',
    ]
    # subject outline
    parts.append(f'<polygon points="{poly_pts(subject)}" fill="#8b949e" fill-opacity="0.10" '
                 f'stroke="#8b949e" stroke-width="1.5" stroke-dasharray="4 3"/>')
    # clip window
    parts.append(f'<polygon points="{poly_pts(window)}" fill="none" '
                 f'stroke="#ffd43b" stroke-width="1.8" stroke-dasharray="6 3"/>')
    # clipped result
    if clipped:
        parts.append(f'<polygon points="{poly_pts(clipped)}" fill="#06d6a0" fill-opacity="0.30" '
                     f'stroke="#06d6a0" stroke-width="2"/>')
        for p in clipped:
            parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="3" fill="#06d6a0"/>')
    for p in subject:
        parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="2.5" fill="#8b949e"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
