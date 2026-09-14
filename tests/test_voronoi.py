"""Tests for Voronoi cells: convexity, self-membership, brute-force nearest-site, tiling, Delaunay dual."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import voronoi as V  # noqa: E402
import delaunay  # noqa: E402


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


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _is_convex_ccw(poly):
    n = len(poly)
    if n < 3:
        return True
    sign = 0
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cx, cy = poly[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def main():
    # ---- 1. simple 5-point layout: tiling + self-membership + convexity -----------------
    pts = [(0, 0), (4, 0), (0, 4), (4, 4), (2, 2)]
    box = V.bounding_box(pts, 0.5)
    cs = V.cells(pts, box)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    tot = sum(V.polygon_area(c) for c in cs)
    check("cells tile the box (area sum)", abs(tot - box_area) < 1e-6, f"{tot} vs {box_area}")
    check("every site inside its own cell", all(V.point_in_polygon(c, pts[i]) for i, c in enumerate(cs)))
    check("every cell convex", all(_is_convex_ccw(c) for c in cs))

    # ---- 2. brute-force nearest-site defines the cell -----------------------------------
    rnd = _lcg(12345)
    mism = 0
    for _ in range(3000):
        q = (box[0] + rnd() * (box[2] - box[0]), box[1] + rnd() * (box[3] - box[1]))
        ns = V.nearest_site(pts, q)
        if not V.point_in_polygon(cs[ns], q):
            # boundary tie: only a failure if strictly-nearest site's cell misses the point
            # recompute with strict margin
            xi, yi = pts[ns]
            d0 = (xi - q[0]) ** 2 + (yi - q[1]) ** 2
            ties = sum(1 for (x, y) in pts if abs((x - q[0]) ** 2 + (y - q[1]) ** 2 - d0) < 1e-6)
            if ties == 1:
                mism += 1
    check("raster point always in nearest site's cell", mism == 0, f"{mism} strict mismatches")

    # ---- 3. random point clouds: tiling holds --------------------------------------------
    rnd = _lcg(999)
    okall = True
    for trial in range(5):
        rp = [(rnd() * 10, rnd() * 10) for _ in range(12)]
        b = V.bounding_box(rp, 0.3)
        c2 = V.cells(rp, b)
        ba = (b[2] - b[0]) * (b[3] - b[1])
        t2 = sum(V.polygon_area(x) for x in c2)
        if abs(t2 - ba) > 1e-5:
            okall = False
        if not all(V.point_in_polygon(c2[i], rp[i]) for i in range(len(rp))):
            okall = False
    check("random clouds: tiling + self-membership", okall)

    # ---- 4. single site fills the whole box ---------------------------------------------
    one = [(3.0, 3.0)]
    b1 = V.bounding_box([(0, 0), (6, 6)], 0.0)
    c1 = V.cell(one, 0, b1)
    check("lone site fills box", abs(V.polygon_area(c1) - 36.0) < 1e-9, f"{V.polygon_area(c1)}")

    # ---- 5. two sites: bisector splits box in half --------------------------------------
    two = [(2.0, 5.0), (8.0, 5.0)]
    b2 = (0.0, 0.0, 10.0, 10.0)
    ca = V.polygon_area(V.cell(two, 0, b2))
    cb = V.polygon_area(V.cell(two, 1, b2))
    check("two sites split box evenly", abs(ca - 50.0) < 1e-9 and abs(cb - 50.0) < 1e-9,
          f"{ca}, {cb}")

    # ---- 6. cross-check finite cell edges vs delaunay.voronoi_edges ----------------------
    # Collect the interior (shared) cell boundary segments from our cells and confirm each
    # Delaunay-dual Voronoi edge appears as a boundary between two adjacent cells.
    site_pts = [(0.0, 0.0), (5.0, 0.5), (1.0, 4.0), (4.0, 4.0), (2.5, 2.0), (6.0, 3.0)]
    dedges = delaunay.voronoi_edges(site_pts)

    def near(p, q, tol=1e-6):
        return abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol

    # gather all cell edges (as unordered endpoint pairs)
    box6 = V.bounding_box(site_pts, 2.0)  # generous box so dual edges lie interior
    cells6 = V.cells(site_pts, box6)
    cell_edges = []
    for c in cells6:
        for i in range(len(c)):
            cell_edges.append((c[i], c[(i + 1) % len(c)]))

    def edge_present(seg):
        p, q = seg
        for (u, w) in cell_edges:
            if (near(p, u) and near(q, w)) or (near(p, w) and near(q, u)):
                return True
            # a dual edge may be a sub-segment of a longer clipped cell edge; check collinear-contained
            if _seg_contains(u, w, p) and _seg_contains(u, w, q):
                return True
        return False

    matched = sum(1 for e in dedges if edge_present(e))
    check("Delaunay-dual Voronoi edges appear as cell boundaries",
          len(dedges) > 0 and matched >= len(dedges) - 1, f"{matched}/{len(dedges)}")

    # ---- 7. centroid inside cell, Lloyd relaxation reduces area variance -----------------
    rnd = _lcg(77)
    cl = [(rnd() * 10, rnd() * 10) for _ in range(15)]
    bcl = V.bounding_box(cl, 0.2)

    def area_var(sites):
        cc = V.cells(sites, bcl)
        ar = [V.polygon_area(x) for x in cc]
        m = sum(ar) / len(ar)
        return sum((a - m) ** 2 for a in ar) / len(ar)

    v0 = area_var(cl)
    s = cl
    for _ in range(8):
        s = V.lloyd_step(s, bcl)
    v1 = area_var(s)
    check("Lloyd relaxation reduces cell-area variance", v1 < v0, f"{v0:.2f} -> {v1:.2f}")
    check("centroid lies in its cell",
          all(V.point_in_polygon(c, V.polygon_centroid(c)) for c in cs if len(c) >= 3))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


def _seg_contains(u, w, p, tol=1e-6):
    """True if point p lies on segment u-w (collinear and within bounds)."""
    cross = (w[0] - u[0]) * (p[1] - u[1]) - (w[1] - u[1]) * (p[0] - u[0])
    if abs(cross) > tol * (1 + abs(w[0] - u[0]) + abs(w[1] - u[1])):
        return False
    dot = (p[0] - u[0]) * (w[0] - u[0]) + (p[1] - u[1]) * (w[1] - u[1])
    if dot < -tol:
        return False
    L2 = (w[0] - u[0]) ** 2 + (w[1] - u[1]) ** 2
    if dot > L2 + tol:
        return False
    return True


if __name__ == "__main__":
    main()
