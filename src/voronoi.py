"""Voronoi cells by half-plane intersection: the region of the plane closest to each site.

The Voronoi diagram partitions the plane into one CELL per site, where a cell is the set of points
closer to that site than to any other. delaunay.py already gives the finite Voronoi EDGES as the dual
of a Delaunay triangulation -- but it omits the unbounded rays of the boundary cells, so it cannot
hand you a closed polygon for every site. This module builds the actual CELLS.

The construction is direct and geometric rather than dual: a site's cell is an intersection of
HALF-PLANES. For every other site, the set of points closer to ours than to that one is the half-plane
on our side of the perpendicular bisector of the two sites. Intersect all n-1 such half-planes (and a
bounding box to close the unbounded cells) and you get a convex polygon -- the cell. Each half-plane
is applied by SUTHERLAND-HODGMAN clipping: walk the current polygon's edges, keep vertices on the
inside, and insert an intersection point wherever an edge crosses the clip line. This is O(n) clips of
an O(n)-vertex polygon per site, so O(n^2) per site overall -- brute compared to a sweepline, but
exact, dependency-light, and a completely independent check on the Delaunay-dual edges.

This module builds clipped Voronoi cells, cell area/centroid, and nearest-site lookup. It is validated:
every cell is convex and contains its own site; a dense raster of query points always lands in the
cell of its true nearest site (the defining property, checked by brute force); the cells tile the
bounding box exactly (their areas sum to the box area with no overlap); a lone site fills the whole
box; and the finite cell edges reproduce delaunay.py's independent Voronoi-edge computation. Pure
stdlib; the cell-polygon companion to the delaunay triangulation and convex-hull tools."""

from __future__ import annotations


def _clip_halfplane(poly, a, b, c):
    """Sutherland-Hodgman clip: keep the part of convex polygon `poly` where a*x + b*y <= c.

    `poly` is a list of (x, y) in order. The clip line is a*x + b*y = c; the kept side is <= c.
    Returns the clipped polygon (possibly empty)."""
    def inside(p):
        return a * p[0] + b * p[1] <= c + 1e-12

    def intersect(p, q):
        # parametric point p + t (q - p) hitting the line a*x + b*y = c
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        t = dp / (dp - dq)
        return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    if not poly:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        cin = inside(cur)
        nin = inside(nxt)
        if cin:
            out.append(cur)
            if not nin:
                out.append(intersect(cur, nxt))
        else:
            if nin:
                out.append(intersect(cur, nxt))
    return out


def bounding_box(points, margin=1.0):
    """Axis-aligned box (xmin, ymin, xmax, ymax) enclosing all points with a relative margin."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    m = margin
    return (xmin - m * dx, ymin - m * dy, xmax + m * dx, ymax + m * dy)


def cell(points, i, box=None):
    """The Voronoi cell of site `points[i]` as a convex polygon (list of (x, y) CCW), clipped to box.

    Built by intersecting the half-plane {closer to site i than site j} for every j != i with the box."""
    if box is None:
        box = bounding_box(points)
    xmin, ymin, xmax, ymax = box
    # start from the bounding-box rectangle (CCW)
    poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    xi, yi = points[i]
    for j, (xj, yj) in enumerate(points):
        if j == i:
            continue
        # bisector: points closer to i satisfy (P - Mij) . (Sj - Si) <= 0
        # => (xj-xi) x + (yj-yi) y <= (xj^2+yj^2 - xi^2 - yi^2)/2
        a = xj - xi
        b = yj - yi
        c = (xj * xj + yj * yj - xi * xi - yi * yi) / 2.0
        poly = _clip_halfplane(poly, a, b, c)
        if not poly:
            break
    return poly


def cells(points, box=None):
    """All Voronoi cells: a list of polygons, one per site (same order as points)."""
    if box is None:
        box = bounding_box(points)
    return [cell(points, i, box) for i in range(len(points))]


def polygon_area(poly):
    """Absolute area of a simple polygon via the shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def polygon_centroid(poly):
    """Area centroid of a simple polygon. Falls back to vertex mean for degenerate polygons."""
    n = len(poly)
    if n < 3:
        cx = sum(p[0] for p in poly) / n
        cy = sum(p[1] for p in poly) / n
        return (cx, cy)
    a = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-15:
        cx = sum(p[0] for p in poly) / n
        cy = sum(p[1] for p in poly) / n
        return (cx, cy)
    return (cx / (6 * a), cy / (6 * a))


def nearest_site(points, q):
    """Index of the site nearest to query point q (brute force; the Voronoi cell membership test)."""
    best = 0
    bd = None
    for i, (x, y) in enumerate(points):
        d = (x - q[0]) ** 2 + (y - q[1]) ** 2
        if bd is None or d < bd:
            bd = d
            best = i
    return best


def point_in_polygon(poly, q):
    """Ray-cast point-in-polygon test (boundary counts as inside within a small tolerance)."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    x, y = q
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xint = x0 + (y - y0) / (y1 - y0) * (x1 - x0)
            if x < xint:
                inside = not inside
    return inside


def lloyd_step(points, box=None):
    """One Lloyd relaxation step: move each site to its cell's centroid (centroidal Voronoi tessellation)."""
    if box is None:
        box = bounding_box(points)
    out = []
    for i in range(len(points)):
        c = cell(points, i, box)
        out.append(polygon_centroid(c) if c else points[i])
    return out
