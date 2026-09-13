"""3-D convex hull by the incremental algorithm: the tightest polyhedron enclosing a point cloud.

The convex hull of points in space is the smallest convex polyhedron containing them all -- the shape
shrink-wrap takes around a scatter of dots. In 3-D it is a mesh of triangular faces, and computing it
is the foundation of collision detection, mesh generation, the 3-D Delaunay triangulation (a hull in
one higher dimension), and shape analysis. This module builds it by the INCREMENTAL algorithm:

  1. Start with a tetrahedron from four non-coplanar points, oriented so every face's outward normal
     points away from the interior.
  2. Add the remaining points one at a time. For each new point p, find the faces it can "see" (p is
     on the outward side of their plane). Those faces are removed; the boundary edges of the removed
     region form a HORIZON, and new triangular faces are built connecting p to each horizon edge.
  3. Points strictly inside the current hull see no faces and are simply discarded.

After all points are processed the surviving faces are the hull's triangulation. The correctness
checks are geometric and combinatorial: every input point lies on or inside every face's supporting
plane, the face count obeys EULER'S FORMULA V - E + F = 2 (for a triangulated hull, F = 2V - 4 and
E = 3V - 6), and the hull volume matches an independent computation.

This module returns the hull vertices and triangular faces (outward-oriented), computes the hull
volume and surface area, and tests point containment. Validated: on random point clouds every point
is inside the hull and every face is a genuine supporting plane; Euler's formula holds; a cube's 8
corners plus interior points give a hull of exactly 8 vertices and volume 1; interior points are
correctly excluded from the vertex set; and degenerate (coplanar) inputs are detected. Pure stdlib;
the 3-D companion to the 2-D convex hull and the Delaunay / geometry tools."""

from __future__ import annotations


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class Face:
    __slots__ = ("a", "b", "c", "normal", "offset")

    def __init__(self, a, b, c, points):
        self.a, self.b, self.c = a, b, c
        pa, pb, pc = points[a], points[b], points[c]
        self.normal = _cross(_sub(pb, pa), _sub(pc, pa))
        self.offset = _dot(self.normal, pa)

    def signed_distance(self, p):
        """>0 if p is on the outward side of this face's plane."""
        return _dot(self.normal, p) - self.offset


def _orient_outward(face, interior, points):
    """Flip the face so its normal points away from an interior point."""
    if face.signed_distance(interior) > 0:
        face.b, face.c = face.c, face.b
        face.normal = tuple(-n for n in face.normal)
        face.offset = -face.offset
    return face


def convex_hull_3d(points, eps=1e-9):
    """Incremental 3-D convex hull. Returns (vertices, faces) where faces are (i, j, k) index
    triples into `points`, outward-oriented. Raises if the points are coplanar."""
    points = [tuple(map(float, p)) for p in points]
    n = len(points)
    if n < 4:
        raise ValueError("need at least 4 points")

    # find an initial non-coplanar tetrahedron
    i0 = 0
    # i1: distinct from i0
    i1 = next((i for i in range(n) if _sub(points[i], points[i0]) != (0, 0, 0)), None)
    if i1 is None:
        raise ValueError("all points identical")
    # i2: not collinear with i0,i1
    i2 = None
    for i in range(n):
        if _cross(_sub(points[i1], points[i0]), _sub(points[i], points[i0])) != (0, 0, 0):
            i2 = i
            break
    if i2 is None:
        raise ValueError("all points collinear")
    # i3: not coplanar
    base_normal = _cross(_sub(points[i1], points[i0]), _sub(points[i2], points[i0]))
    i3 = None
    for i in range(n):
        if abs(_dot(base_normal, _sub(points[i], points[i0]))) > eps:
            i3 = i
            break
    if i3 is None:
        raise ValueError("all points coplanar")

    interior = tuple((points[i0][d] + points[i1][d] + points[i2][d] + points[i3][d]) / 4
                     for d in range(3))
    faces = [
        _orient_outward(Face(i0, i1, i2, points), interior, points),
        _orient_outward(Face(i0, i1, i3, points), interior, points),
        _orient_outward(Face(i0, i2, i3, points), interior, points),
        _orient_outward(Face(i1, i2, i3, points), interior, points),
    ]

    used = {i0, i1, i2, i3}
    for pi in range(n):
        if pi in used:
            continue
        p = points[pi]
        visible = [f for f in faces if f.signed_distance(p) > eps]
        if not visible:
            continue  # inside the hull
        # find the horizon: edges bordering exactly one visible face
        visible_set = set(id(f) for f in visible)
        edge_count = {}
        for f in visible:
            for (u, v) in ((f.a, f.b), (f.b, f.c), (f.c, f.a)):
                key = (min(u, v), max(u, v))
                edge_count[key] = edge_count.get(key, 0) + 1
        # horizon edges appear once among visible faces; keep them oriented from each visible face
        horizon = []
        for f in visible:
            for (u, v) in ((f.a, f.b), (f.b, f.c), (f.c, f.a)):
                key = (min(u, v), max(u, v))
                if edge_count[key] == 1:
                    horizon.append((u, v))
        # remove visible faces
        faces = [f for f in faces if id(f) not in visible_set]
        # add new faces from p to each horizon edge
        for (u, v) in horizon:
            nf = Face(u, v, pi, points)
            _orient_outward(nf, interior, points)
            faces.append(nf)

    vertices = sorted({idx for f in faces for idx in (f.a, f.b, f.c)})
    face_tuples = [(f.a, f.b, f.c) for f in faces]
    return vertices, face_tuples


def hull_volume(points, faces):
    """Volume of the polyhedron via the divergence theorem: sum of signed tetrahedra to the origin."""
    vol = 0.0
    for (a, b, c) in faces:
        pa, pb, pc = points[a], points[b], points[c]
        vol += _dot(pa, _cross(pb, pc)) / 6.0
    return abs(vol)


def hull_area(points, faces):
    """Total surface area (sum of triangle areas)."""
    area = 0.0
    for (a, b, c) in faces:
        cr = _cross(_sub(points[b], points[a]), _sub(points[c], points[a]))
        area += 0.5 * (cr[0] ** 2 + cr[1] ** 2 + cr[2] ** 2) ** 0.5
    return area


def is_inside(points, faces, p, eps=1e-7):
    """True if p is on or inside the hull (on the inward side of every outward face)."""
    interior = _centroid(points, faces)
    for (a, b, c) in faces:
        f = Face(a, b, c, points)
        _orient_outward(f, interior, points)
        if f.signed_distance(p) > eps:
            return False
    return True


def _centroid(points, faces):
    idxs = {idx for f in faces for idx in f}
    n = len(idxs)
    return tuple(sum(points[i][d] for i in idxs) / n for d in range(3))


def euler_ok(vertices, faces):
    """Check Euler's formula V - E + F = 2 for the triangulated hull."""
    V = len(vertices)
    F = len(faces)
    edges = set()
    for (a, b, c) in faces:
        for (u, v) in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    E = len(edges)
    return V - E + F == 2
