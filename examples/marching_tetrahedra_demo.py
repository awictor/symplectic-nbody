"""Demo: marching tetrahedra meshing a 3-D isosurface -- a two-blob metaball union.

Meshes the isosurface of a metaball field (two Gaussian-like blobs) into triangles, reports how the
sphere surface area converges to the analytic 4 pi r^2, and draws the blob mesh in an orthographic
projection with depth shading.

    python examples/marching_tetrahedra_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from marching_tetrahedra import isosurface, total_area, triangle_area  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Marching tetrahedra: meshing a 3-D isosurface without a 256-case table\n")

    # --- convergence check on a sphere r = 1 ---
    R = 1.0

    def sphere(x, y, z):
        return x * x + y * y + z * z - R * R

    target = 4 * math.pi * R * R
    print(f"  Sphere r=1: mesh surface area -> analytic 4 pi r^2 = {target:.4f}")
    print(f"    {'grid':>8}{'triangles':>12}{'area':>12}{'error':>10}")
    for n in [8, 16, 32]:
        t = isosurface(sphere, -1.3, 1.3, -1.3, 1.3, -1.3, 1.3, level=0.0, nx=n, ny=n, nz=n)
        a = total_area(t)
        print(f"    {n:>8}{len(t):>12}{a:>12.4f}{abs(a - target) / target:>9.2%}")

    # --- the blob to render: two metaballs, surface where field = 1 ---
    def metaballs(x, y, z):
        d1 = (x - 0.6) ** 2 + y * y + z * z
        d2 = (x + 0.6) ** 2 + y * y + z * z
        return 1.0 / (d1 + 0.05) + 1.0 / (d2 + 0.05)

    tris = isosurface(metaballs, -2.0, 2.0, -1.5, 1.5, -1.5, 1.5, level=2.5,
                      nx=28, ny=24, nz=24)
    print(f"\n  Two-blob metaball union at level 2.5: {len(tris)} triangles, "
          f"area {total_area(tris):.3f}")

    _svg(os.path.join(outdir, "marching_tetrahedra.svg"), tris)
    print(f"\n  Watertight by construction: adjacent tetrahedra share whole faces and interpolate")
    print(f"  identically on shared edges, so the mesh has no holes -- no ambiguous cases.")
    print(f"\n  wrote {os.path.join(outdir, 'marching_tetrahedra.svg')}")


def _svg(path, tris, width=760, height=460):
    # orthographic projection: view along a tilted axis, simple painter's algorithm by depth.
    # rotate points a bit for a 3/4 view
    ca, sa = math.cos(0.6), math.sin(0.6)   # yaw
    cb, sb = math.cos(0.5), math.sin(0.5)   # pitch

    def project(p):
        x, y, z = p
        # yaw about y
        x1 = ca * x + sa * z
        z1 = -sa * x + ca * z
        # pitch about x
        y1 = cb * y - sb * z1
        depth = sb * y + cb * z1
        return x1, y1, depth

    faces = []
    for tri in tris:
        proj = [project(p) for p in tri]
        depth = sum(p[2] for p in proj) / 3.0
        faces.append((depth, proj))
    faces.sort(key=lambda f: f[0])  # far to near

    xs = [p[0] for _, pr in faces for p in pr]
    ys = [p[1] for _, pr in faces for p in pr]
    if not xs:
        xs, ys = [0, 1], [0, 1]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    depths = [d for d, _ in faces]
    dmin, dmax = (min(depths), max(depths)) if depths else (0.0, 1.0)
    pad = 30

    def sx(x):
        return pad + (width - 2 * pad) * (x - minx) / (maxx - minx + 1e-9)

    def sy(y):
        return height - pad - (height - 2 * pad - 20) * (y - miny) / (maxy - miny + 1e-9)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Marching tetrahedra: a two-blob metaball isosurface ({len(tris)} triangles)</text>',
    ]
    for depth, pr in faces:
        t = (depth - dmin) / (dmax - dmin + 1e-9)  # 0 far .. 1 near
        # depth shade from dark blue to bright cyan
        r = int(40 + 20 * t)
        g = int(90 + 130 * t)
        b = int(120 + 90 * t)
        pts = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pr)
        parts.append(f'<polygon points="{pts}" fill="rgb({r},{g},{b})" '
                     f'stroke="#0d1117" stroke-width="0.3"/>')
    parts.append(f'<text x="20" y="{height-16}" fill="#8b949e" font-size="10">'
                 f'orthographic 3/4 view, painter&#39;s algorithm, depth-shaded</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
