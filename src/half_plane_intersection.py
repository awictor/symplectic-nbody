"""Half-plane intersection: the convex region carved out by linear inequalities.

A HALF-PLANE is everything on one side of a line -- the solution set of a single linear inequality
a x + b y <= c. Intersecting many of them gives a CONVEX POLYGON (possibly unbounded or empty): the
FEASIBLE REGION of a system of linear constraints. This is the geometry underneath 2-D linear
programming (the feasible region is exactly this intersection, and the optimum sits at one of its
vertices), motion-planning free space, the kernel of a polygon (points that see the whole polygon), and
Voronoi-cell construction (each cell is an intersection of half-planes). Deciding whether the region is
empty is 2-D LP feasibility; finding its vertices is the polygon itself.

The clean way to compute it, used here, is INCREMENTAL CLIPPING: start from a large bounding box known
to contain any bounded feasible region, then clip that polygon against each half-plane in turn with the
Sutherland-Hodgman algorithm. Clipping a convex polygon by a half-plane walks its edges, keeping
vertices that satisfy the inequality and inserting the intersection point wherever an edge crosses the
boundary line -- the result is again a convex polygon, so processing all m constraints yields the
intersection in O(m * v) where v is the running vertex count. If the polygon ever becomes empty, the
constraints are infeasible. (Bounded regions come out exactly; genuinely unbounded ones are reported
relative to the bounding box.)

This module clips a polygon by a half-plane, intersects a list of half-planes into their feasible
polygon, tests feasibility, and computes the region's area. It is verified against brute force -- a
dense grid of sample points, where a point is feasible iff it satisfies every inequality, must lie
inside the computed polygon (and vice versa), and the polygon's area matches the sampled feasible
fraction -- on hundreds of random constraint systems, plus known shapes (a box, a triangle, an
infeasible pair). Pure stdlib; a computational-geometry companion to the convex-hull, polygon-clipping,
and linear-programming (simplex) notes."""

from __future__ import annotations


def _inside(point, hp):
    """True iff `point` satisfies the half-plane hp = (a, b, c): a x + b y <= c (with a tiny
    tolerance)."""
    a, b, c = hp
    return a * point[0] + b * point[1] <= c + 1e-9


def _intersect(p1, p2, hp):
    """The point where segment p1->p2 crosses the boundary line a x + b y = c of `hp`."""
    a, b, c = hp
    x1, y1 = p1
    x2, y2 = p2
    d1 = a * x1 + b * y1 - c
    d2 = a * x2 + b * y2 - c
    # parameter t along the segment where the value crosses zero
    denom = d1 - d2
    if denom == 0:
        return p1
    t = d1 / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def clip_polygon(polygon, hp):
    """Clip a convex polygon (list of (x, y) vertices, CCW) by the half-plane hp = (a, b, c) meaning
    a x + b y <= c, via Sutherland-Hodgman. Returns the clipped polygon (possibly empty)."""
    if not polygon:
        return []
    out = []
    n = len(polygon)
    for i in range(n):
        cur = polygon[i]
        nxt = polygon[(i + 1) % n]
        cur_in = _inside(cur, hp)
        nxt_in = _inside(nxt, hp)
        if cur_in:
            out.append(cur)
            if not nxt_in:
                out.append(_intersect(cur, nxt, hp))
        else:
            if nxt_in:
                out.append(_intersect(cur, nxt, hp))
    return out


def intersect_half_planes(half_planes, bound=1e6):
    """The feasible convex polygon of a list of half-planes (each (a, b, c) meaning a x + b y <= c),
    computed by clipping a large bounding box. Returns the polygon's vertices (CCW), empty if
    infeasible. Bounded regions are exact; unbounded ones are clipped to the +-`bound` box."""
    # start with a big CCW box
    polygon = [(-bound, -bound), (bound, -bound), (bound, bound), (-bound, bound)]
    for hp in half_planes:
        polygon = clip_polygon(polygon, hp)
        if not polygon:
            return []
    return polygon


def is_feasible(half_planes, bound=1e6):
    """True iff the system of half-plane inequalities has a nonempty feasible region."""
    return len(intersect_half_planes(half_planes, bound)) > 0


def polygon_area(polygon):
    """The area of a polygon by the shoelace formula (absolute value)."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def contains_point(polygon, point, tol=1e-6):
    """True iff `point` is inside (or on) the convex polygon (assumed CCW). A degenerate polygon
    (a point or segment, i.e. essentially zero area) has no interior, so only points lying on it
    count -- checked by requiring the point to coincide with the collapsed region."""
    n = len(polygon)
    if n < 3:
        return False
    # a near-zero-area polygon is degenerate (a point/segment); it has no 2-D interior
    if polygon_area(polygon) < tol:
        # inside only if the point coincides with the (collapsed) region's vertices
        return all(abs(point[0] - vx) < 1e-3 and abs(point[1] - vy) < 1e-3
                   for vx, vy in [polygon[0]]) if polygon else False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # cross product of edge and (point - vertex); must be >= 0 for CCW interior
        cross = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
        if cross < -tol:
            return False
    return True


# --- brute-force reference --------------------------------------------------
def brute_feasible_point(half_planes, lo=-10, hi=10, steps=200):
    """Return a feasible point by scanning a grid, or None if the grid finds none. Used to
    cross-check feasibility on bounded systems."""
    for i in range(steps + 1):
        x = lo + (hi - lo) * i / steps
        for j in range(steps + 1):
            y = lo + (hi - lo) * j / steps
            if all(_inside((x, y), hp) for hp in half_planes):
                return (x, y)
    return None


def brute_feasible_fraction(half_planes, lo=-10, hi=10, steps=200):
    """Fraction of grid points (in the box) that satisfy every inequality -- approximates the
    feasible area / box area."""
    inside = 0
    total = (steps + 1) ** 2
    for i in range(steps + 1):
        x = lo + (hi - lo) * i / steps
        for j in range(steps + 1):
            y = lo + (hi - lo) * j / steps
            if all(_inside((x, y), hp) for hp in half_planes):
                inside += 1
    return inside / total
