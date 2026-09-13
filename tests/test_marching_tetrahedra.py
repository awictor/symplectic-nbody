"""Tests for marching tetrahedra: vertices on the surface, sphere area -> 4 pi r^2, plane exact."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from marching_tetrahedra import (  # noqa: E402
    isosurface,
    tetra_triangles,
    triangle_area,
    total_area,
)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    R = 1.0

    def sphere(x, y, z):
        return x * x + y * y + z * z - R * R

    # ---- 1. every triangle vertex lies on the isosurface (f ~ 0) ------------------------
    tris = isosurface(sphere, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, level=0.0, nx=16, ny=16, nz=16)
    check("sphere produced triangles", len(tris) > 0, f"{len(tris)}")
    # vertices lie on the LINEAR interpolant's zero, so the true nonlinear field deviates by
    # O(dx^2) along each edge -- bound the deviation by the grid spacing, not by zero.
    dx = 3.0 / 16
    max_dev = 0.0
    for tri in tris:
        for (x, y, z) in tri:
            max_dev = max(max_dev, abs(sphere(x, y, z)))
    check("all vertices lie on the surface (O(dx^2))", max_dev < dx * dx,
          f"max |f| = {max_dev:.2e}, dx^2 = {dx*dx:.2e}")

    # ---- 2. all vertices inside the sampled box -----------------------------------------
    inside = all(-1.5 - 1e-9 <= c <= 1.5 + 1e-9 for tri in tris for pt in tri for c in pt)
    check("all vertices within the box", inside)

    # ---- 3. sphere surface area converges to 4 pi r^2 -----------------------------------
    target = 4 * math.pi * R * R
    areas = []
    for n in [10, 20, 40]:
        t = isosurface(sphere, -1.4, 1.4, -1.4, 1.4, -1.4, 1.4, level=0.0, nx=n, ny=n, nz=n)
        areas.append(total_area(t))
    # marching tetrahedra slightly under-estimates a sphere but converges upward toward 4 pi r^2
    err_coarse = abs(areas[0] - target) / target
    err_fine = abs(areas[-1] - target) / target
    check("sphere area near 4 pi r^2 at fine grid", err_fine < 0.03,
          f"area {areas[-1]:.4f} vs {target:.4f} ({err_fine:.3%})")
    check("area error shrinks with refinement", err_fine < err_coarse,
          f"{err_coarse:.3%} -> {err_fine:.3%}")

    # ---- 4. a flat plane f = z - 0.3 reproduces its area exactly -------------------------
    def plane(x, y, z):
        return z - 0.3

    ptris = isosurface(plane, 0.0, 2.0, 0.0, 3.0, -1.0, 1.0, level=0.0, nx=8, ny=8, nz=8)
    # the plane cuts the box in a 2 x 3 rectangle -> area 6
    check("plane area is exact 6.0", abs(total_area(ptris) - 6.0) < 1e-9,
          f"{total_area(ptris)}")

    # ---- 5. a plane's vertices all have z = 0.3 -----------------------------------------
    zconst = all(abs(z - 0.3) < 1e-9 for tri in ptris for (_, _, z) in tri)
    check("plane vertices all at z=0.3", zconst)

    # ---- 6. per-tetrahedron cases -------------------------------------------------------
    unit = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # all below -> no triangles
    check("all-below tet -> 0 triangles",
          tetra_triangles(unit, [-1, -1, -1, -1], 0.0) == [])
    # all above -> no triangles
    check("all-above tet -> 0 triangles",
          tetra_triangles(unit, [1, 1, 1, 1], 0.0) == [])
    # one below -> 1 triangle
    check("one-below tet -> 1 triangle",
          len(tetra_triangles(unit, [-1, 1, 1, 1], 0.0)) == 1)
    # two below -> 2 triangles (quad)
    check("two-below tet -> 2 triangles",
          len(tetra_triangles(unit, [-1, -1, 1, 1], 0.0)) == 2)

    # ---- 7. midpoint interpolation: symmetric values put the cut at the edge midpoint ---
    tri = tetra_triangles(unit, [-1, 1, 1, 1], 0.0)[0]
    # lone corner 0 at origin; edges to (1,0,0),(0,1,0),(0,0,1) each cut at their midpoint
    expected = {(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)}
    got = {tuple(round(c, 6) for c in p) for p in tri}
    check("symmetric cut lands at edge midpoints", got == expected, f"{got}")

    # ---- 8. triangle area sanity --------------------------------------------------------
    unit_tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    check("unit right triangle area = 0.5", abs(triangle_area(unit_tri) - 0.5) < 1e-12)

    # ---- 9. off-center sphere: still all on surface -------------------------------------
    def sphere2(x, y, z):
        return (x - 0.4) ** 2 + (y + 0.3) ** 2 + z ** 2 - 0.64  # r = 0.8

    t2 = isosurface(sphere2, -0.6, 1.4, -1.3, 0.7, -1.0, 1.0, level=0.0, nx=16, ny=16, nz=16)
    dev = max(abs(sphere2(*pt)) for tri in t2 for pt in tri) if t2 else 1.0
    dx2 = 2.0 / 16
    check("off-center sphere vertices on surface (O(dx^2))", dev < dx2 * dx2,
          f"{dev:.2e} vs dx^2 {dx2*dx2:.2e}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
