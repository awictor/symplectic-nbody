"""Tests for 3-D convex hull: all points inside, Euler's formula, cube/tetra volumes, exclusion."""

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
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    # ---- 1. random clouds: all points inside, Euler holds -------------------------------
    rng = _lcg(2024)
    all_inside = euler_all = True
    for _ in range(30):
        n = 8 + int(rng() * 20)
        pts = [(rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1) for _ in range(n)]
        try:
            verts, faces = convex_hull_3d(pts)
        except ValueError:
            continue
        for p in pts:
            if not is_inside(pts, faces, p, eps=1e-6):
                all_inside = False
        if not euler_ok(verts, faces):
            euler_all = False
    check("all points inside their hull (30 clouds)", all_inside)
    check("Euler V - E + F = 2 holds", euler_all)

    # ---- 2. triangulated hull satisfies F = 2V - 4 --------------------------------------
    rng = _lcg(77)
    ok = True
    for _ in range(20):
        n = 10 + int(rng() * 15)
        pts = [(rng(), rng(), rng()) for _ in range(n)]
        verts, faces = convex_hull_3d(pts)
        if len(faces) != 2 * len(verts) - 4:
            ok = False
    check("triangulated hull has F = 2V - 4 faces", ok)

    # ---- 3. unit cube: 8 vertices, volume 1, area 6 -------------------------------------
    cube = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    # add interior points that must NOT become hull vertices
    interior_pts = [(0.5, 0.5, 0.5), (0.3, 0.7, 0.2), (0.6, 0.4, 0.8)]
    pts = cube + interior_pts
    verts, faces = convex_hull_3d(pts)
    check("cube hull has 8 vertices", len(verts) == 8, f"{len(verts)}")
    check("cube hull volume = 1", abs(hull_volume(pts, faces) - 1.0) < 1e-9,
          f"{hull_volume(pts, faces):.6f}")
    check("cube hull surface area = 6", abs(hull_area(pts, faces) - 6.0) < 1e-9,
          f"{hull_area(pts, faces):.6f}")
    check("interior points excluded from vertices", all(i < 8 for i in verts))

    # ---- 4. regular tetrahedron volume --------------------------------------------------
    # tetra with vertices; volume = |det| / 6
    tetra = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    verts, faces = convex_hull_3d(tetra)
    check("tetra: 4 vertices, 4 faces", len(verts) == 4 and len(faces) == 4)
    check("tetra volume = 1/6", abs(hull_volume(tetra, faces) - 1 / 6) < 1e-9,
          f"{hull_volume(tetra, faces):.6f}")

    # ---- 5. points on a sphere: all are hull vertices -----------------------------------
    rng = _lcg(7)
    sphere = []
    for _ in range(20):
        # random point on unit sphere
        while True:
            x, y, z = rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1
            r = math.sqrt(x * x + y * y + z * z)
            if 0.1 < r <= 1:
                sphere.append((x / r, y / r, z / r))
                break
    verts, faces = convex_hull_3d(sphere)
    check("all sphere points are hull vertices", len(verts) == len(sphere), f"{len(verts)}/{len(sphere)}")
    check("sphere hull Euler holds", euler_ok(verts, faces))

    # ---- 6. containment test ------------------------------------------------------------
    cube = [(x, y, z) for x in (0, 2) for y in (0, 2) for z in (0, 2)]
    verts, faces = convex_hull_3d(cube + [(1, 1, 1)])
    check("center is inside cube", is_inside(cube, faces, (1, 1, 1)))
    check("far point is outside cube", not is_inside(cube, faces, (5, 5, 5)))
    check("corner is on the hull", is_inside(cube, faces, (0, 0, 0)))

    # ---- 7. every face is a supporting plane (all points on inward side) ----------------
    rng = _lcg(11)
    pts = [(rng() * 3, rng() * 3, rng() * 3) for _ in range(25)]
    verts, faces = convex_hull_3d(pts)
    from convex_hull_3d import Face, _orient_outward, _centroid
    interior = _centroid(pts, faces)
    ok = True
    for (a, b, c) in faces:
        f = _orient_outward(Face(a, b, c, pts), interior, pts)
        for p in pts:
            if f.signed_distance(p) > 1e-6:
                ok = False
    check("every face is a supporting plane", ok)

    # ---- 8. degenerate inputs ------------------------------------------------------------
    coplanar = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0.5, 0.5, 0)]
    try:
        convex_hull_3d(coplanar)
        check("coplanar points raise", False)
    except ValueError:
        check("coplanar points raise", True)
    try:
        convex_hull_3d([(0, 0, 0), (1, 1, 1)])
        check("too few points raise", False)
    except ValueError:
        check("too few points raise", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
