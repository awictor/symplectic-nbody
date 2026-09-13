"""Demo: 3-D convex hull by the incremental algorithm.

Builds the convex hull of a random point cloud, verifies it (Euler's formula, all points inside),
reports volume and area, and draws the hull's triangular faces in an isometric projection.

    python examples/convex_hull_3d_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convex_hull_3d import (  # noqa: E402
    convex_hull_3d,
    hull_volume,
    hull_area,
    is_inside,
    euler_ok,
)


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("3-D convex hull: shrink-wrap around a point cloud (incremental algorithm)\n")

    rng = _lcg(2024)
    n = 40
    pts = [(rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1) for _ in range(n)]
    verts, faces = convex_hull_3d(pts)

    print(f"  {n} random points in a cube [-1,1]^3.")
    print(f"  Hull: {len(verts)} vertices, {len(faces)} triangular faces.")
    print(f"  Euler's formula V - E + F = 2 holds: {euler_ok(verts, faces)}")
    print(f"  All points inside the hull: {all(is_inside(pts, faces, p, eps=1e-6) for p in pts)}")
    print(f"  Hull volume {hull_volume(pts, faces):.4f}, surface area {hull_area(pts, faces):.4f}")
    print(f"  Interior points discarded: {n - len(verts)} of {n} were inside.\n")

    # a few Platonic-ish checks
    cube = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    _, cfaces = convex_hull_3d(cube + [(0.5, 0.5, 0.5)])
    print(f"  Unit cube (8 corners + center): hull volume {hull_volume(cube+[(0.5,)*3], cfaces):.3f}, "
          f"area {hull_area(cube+[(0.5,)*3], cfaces):.3f} (expect 1.0 and 6.0)")

    print("\n  The incremental algorithm adds points one at a time, deleting the faces each new point")
    print("  can 'see' and stitching new triangles across the horizon -- interior points see nothing.")

    _svg(os.path.join(outdir, "convex_hull_3d.svg"), pts, verts, faces)
    print(f"\n  wrote {os.path.join(outdir, 'convex_hull_3d.svg')}")


def _svg(path, pts, verts, faces, width=760, height=430):
    # isometric-ish projection
    def project(p):
        x, y, z = p
        # rotate for a pleasant view
        ax = 0.5
        ay = 0.9
        # simple oblique: rotate about y then x
        xr = x * math.cos(ay) + z * math.sin(ay)
        zr = -x * math.sin(ay) + z * math.cos(ay)
        yr = y * math.cos(ax) - zr * math.sin(ax)
        return (xr, yr)

    proj = [project(p) for p in pts]
    xs = [q[0] for q in proj]
    ys = [q[1] for q in proj]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ox, oy, ow, oh = 60, 60, width - 120, height - 120

    def sx(q):
        return ox + ow * (q[0] - xmin) / (xmax - xmin or 1)

    def sy(q):
        return oy + oh * (1 - (q[1] - ymin) / (ymax - ymin or 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'3-D convex hull of {len(pts)} points: {len(verts)} vertices, {len(faces)} faces</text>',
    ]
    # depth-sort faces (painter's algorithm) by average projected z-proxy (use original z)
    def face_depth(f):
        return sum(pts[i][2] + pts[i][0] for i in f) / 3  # crude depth key
    for f in sorted(faces, key=face_depth):
        poly = " ".join(f"{sx(proj[i]):.1f},{sy(proj[i]):.1f}" for i in f)
        parts.append(f'<polygon points="{poly}" fill="#06d6a0" fill-opacity="0.18" '
                     f'stroke="#06d6a0" stroke-width="0.8"/>')
    # points
    vset = set(verts)
    for i, q in enumerate(proj):
        col = "#ffd43b" if i in vset else "#8b949e"
        r = 3 if i in vset else 1.6
        parts.append(f'<circle cx="{sx(q):.1f}" cy="{sy(q):.1f}" r="{r}" fill="{col}"/>')
    parts.append('<text x="20" y="%d" fill="#8b949e" font-size="10">'
                 'yellow = hull vertices, gray = interior points, green = faces</text>' % (height - 15))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
