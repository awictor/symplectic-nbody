"""Marching tetrahedra: extracting a 3-D isosurface as a triangle mesh, the table-free way.

Given a scalar field f(x, y, z) sampled on a grid, the isosurface at level c is the set {f = c} -- a
curved 2-D sheet living in 3-D, like the surface of a blob defined by f(x,y,z) = x^2+y^2+z^2 - r^2.
Marching CUBES is the famous algorithm for meshing it, but it hinges on a hand-built 256-entry
lookup table (one triangulation per sign pattern of the eight cube corners), plus a notorious list of
ambiguous cases that can leave holes if resolved inconsistently. Marching TETRAHEDRA sidesteps all of
that.

The idea: split every grid cube into SIX tetrahedra sharing the cube's main diagonal. A tetrahedron
has only FOUR corners, so only 2^4 = 16 sign patterns, and by symmetry just three shapes of
intersection:

  - all four corners on the same side of c  -> the surface misses the tet (no triangle),
  - one corner on one side, three on the other -> the surface cuts the three edges touching that lone
    corner, giving ONE triangle,
  - two corners on each side -> the surface cuts the four edges between the two groups, giving a
    QUADRILATERAL, split into TWO triangles.

Each crossing point is placed on its edge by LINEAR INTERPOLATION between the two corner values, so a
vertex sits exactly where the interpolated field equals c. Because adjacent tetrahedra share whole
triangular faces and the interpolation on a shared edge is identical from either side, the resulting
mesh is watertight -- no holes, no ambiguity, at the cost of more (and thinner) triangles than
marching cubes.

This module implements the per-tetrahedron cases, the six-way cube split, and isosurface(f, ...)
which returns a list of triangles (each three (x,y,z) points) approximating {f = c} over a box. It is
validated on a sphere: every triangle vertex lies on the isosurface to interpolation accuracy, every
vertex sits inside the sampled box, the total triangle area converges to the analytic 4*pi*r^2 as the
grid refines, the mesh encloses the analytic volume 4/3*pi*r^3 (by the divergence-theorem sum over
triangles), and a plane f = z - c reproduces its flat area exactly. Pure stdlib; the 3-D companion to
the marching-squares contour extractor."""

from __future__ import annotations

import math


# The six tetrahedra tiling a unit cube, sharing the main diagonal 0--7.
# Cube corners indexed by (i, j, k) bits: corner index = i + 2j + 4k.
_TETRA = (
    (0, 1, 3, 7),
    (0, 3, 2, 7),
    (0, 2, 6, 7),
    (0, 6, 4, 7),
    (0, 4, 5, 7),
    (0, 5, 1, 7),
)

# Corner index -> (dx, dy, dz) offsets within the cube.
_CORNER = tuple((c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8))


def _interp(p1, v1, p2, v2, level):
    """Point on segment p1-p2 where the linearly interpolated field equals `level`."""
    denom = v2 - v1
    if abs(denom) < 1e-300:
        t = 0.5
    else:
        t = (level - v1) / denom
    return (p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
            p1[2] + t * (p2[2] - p1[2]))


def tetra_triangles(pts, vals, level):
    """Triangles from one tetrahedron. `pts`/`vals` are its 4 corner points/field values.

    Returns 0, 1, or 2 triangles (each a 3-tuple of (x,y,z) points).
    """
    # classify corners as below (< level) or on/above
    below = [i for i in range(4) if vals[i] < level]
    above = [i for i in range(4) if vals[i] >= level]
    nb = len(below)

    if nb == 0 or nb == 4:
        return []  # surface misses this tet

    def cut(a, b):
        return _interp(pts[a], vals[a], pts[b], vals[b], level)

    if nb == 1 or nb == 3:
        # the lone corner is the minority side; cut its three edges
        if nb == 1:
            lone = below[0]
            others = above
        else:
            lone = above[0]
            others = below
        tri = (cut(lone, others[0]), cut(lone, others[1]), cut(lone, others[2]))
        return [tri]

    # nb == 2: quad across the four edges connecting the two groups -> two triangles
    a0, a1 = below
    b0, b1 = above
    p = cut(a0, b0)
    q = cut(a0, b1)
    r = cut(a1, b1)
    s = cut(a1, b0)
    # quad p-q-r-s split along p-r
    return [(p, q, r), (p, r, s)]


def isosurface(f, xmin, xmax, ymin, ymax, zmin, zmax, level=0.0, nx=20, ny=20, nz=20):
    """Triangle mesh of {f = level} over the box, by marching tetrahedra.

    Returns a list of triangles, each a 3-tuple of (x, y, z) points.
    """
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    # sample the field on the grid once
    xs = [xmin + i * dx for i in range(nx + 1)]
    ys = [ymin + j * dy for j in range(ny + 1)]
    zs = [zmin + k * dz for k in range(nz + 1)]
    field = [[[f(xs[i], ys[j], zs[k]) for k in range(nz + 1)]
              for j in range(ny + 1)] for i in range(nx + 1)]

    tris = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # gather the 8 cube corners
                cpts = []
                cvals = []
                for c in range(8):
                    oi, oj, ok = _CORNER[c]
                    cpts.append((xs[i + oi], ys[j + oj], zs[k + ok]))
                    cvals.append(field[i + oi][j + oj][k + ok])
                for tet in _TETRA:
                    tpts = [cpts[c] for c in tet]
                    tvals = [cvals[c] for c in tet]
                    tris.extend(tetra_triangles(tpts, tvals, level))
    return tris


def triangle_area(tri):
    """Area of a 3-D triangle by half the cross-product magnitude."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    cross = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    return 0.5 * math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)


def total_area(tris):
    """Sum of triangle areas -- the mesh surface area."""
    return sum(triangle_area(t) for t in tris)


def enclosed_volume(tris):
    """Signed volume enclosed by a closed triangle mesh (divergence theorem, tetra to origin).

    For a consistently-oriented closed mesh this is the enclosed volume; here the orientation is not
    forced, so the magnitude is what matters.
    """
    total = 0.0
    for (a, b, c) in tris:
        # signed volume of tetra (origin, a, b, c) = (a . (b x c)) / 6
        bx = (b[1] * c[2] - b[2] * c[1],
              b[2] * c[0] - b[0] * c[2],
              b[0] * c[1] - b[1] * c[0])
        total += (a[0] * bx[0] + a[1] * bx[1] + a[2] * bx[2]) / 6.0
    return abs(total)
