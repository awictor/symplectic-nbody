"""Tests for simplicial homology: known Betti/torsion of standard spaces, d^2=0, Euler-Poincare."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simplicial_homology import (  # noqa: E402
    homology,
    boundary_matrix,
    euler_characteristic,
    euler_from_betti,
    verify_boundary_squared_zero,
    circle, disk, sphere, figure_eight, torus, projective_plane,
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


def _betti_list(betti):
    return [betti[k] for k in sorted(betti)]


def main():
    # ---- 1. circle S^1: Betti (1, 1), no torsion ----------------------------------------
    b, t = homology(circle())
    check("circle Betti (1,1)", _betti_list(b) == [1, 1], f"{_betti_list(b)}")
    check("circle no torsion", all(not t[k] for k in t))

    # ---- 2. disk: contractible, Betti (1, 0, 0) -----------------------------------------
    b, t = homology(disk())
    check("disk Betti (1,0,0)", _betti_list(b) == [1, 0, 0], f"{_betti_list(b)}")

    # ---- 3. sphere S^2: Betti (1, 0, 1) -------------------------------------------------
    b, t = homology(sphere())
    check("sphere Betti (1,0,1)", _betti_list(b) == [1, 0, 1], f"{_betti_list(b)}")
    check("sphere no torsion", all(not t[k] for k in t))

    # ---- 4. figure-eight (wedge of 2 circles): Betti (1, 2) -----------------------------
    b, t = homology(figure_eight())
    check("figure-eight Betti (1,2)", _betti_list(b) == [1, 2], f"{_betti_list(b)}")

    # ---- 5. torus T^2: Betti (1, 2, 1) --------------------------------------------------
    b, t = homology(torus())
    check("torus Betti (1,2,1)", _betti_list(b) == [1, 2, 1], f"{_betti_list(b)}")
    check("torus no torsion (orientable)", all(not t[k] for k in t))

    # ---- 6. projective plane RP^2: Betti (1,0,0) with Z/2 torsion in H_1 ----------------
    b, t = homology(projective_plane())
    check("RP^2 Betti (1,0,0)", _betti_list(b) == [1, 0, 0], f"{_betti_list(b)}")
    check("RP^2 has Z/2 torsion in H_1", t[1] == [2], f"{t}")

    # ---- 7. d_{k-1} d_k = 0 for every complex -------------------------------------------
    for name, cx in [("circle", circle()), ("disk", disk()), ("sphere", sphere()),
                     ("torus", torus()), ("RP2", projective_plane())]:
        check(f"{name}: d^2 = 0", verify_boundary_squared_zero(cx))

    # ---- 8. Euler characteristic = alternating simplex count = alternating Betti --------
    for name, cx, expected_chi in [("circle", circle(), 0), ("disk", disk(), 1),
                                   ("sphere", sphere(), 2), ("torus", torus(), 0),
                                   ("RP2", projective_plane(), 1), ("fig8", figure_eight(), -1)]:
        chi = euler_characteristic(cx)
        b, t = homology(cx)
        chi_b = euler_from_betti(b)
        check(f"{name}: chi = {expected_chi} = alt-Betti",
              chi == expected_chi and chi == chi_b, f"chi={chi}, altBetti={chi_b}")

    # ---- 9. boundary matrix of a single edge: head - tail -------------------------------
    # edge (0,1): d = v1 - v0
    M = boundary_matrix([(0, 1)], [(0,), (1,)])
    check("edge boundary = v1 - v0", M == [[-1], [1]], f"{M}")

    # ---- 10. boundary matrix of a triangle: signed edge sum -----------------------------
    # triangle (0,1,2): faces (1,2) - (0,2) + (0,1)
    tri = [(0, 1, 2)]
    edges = [(0, 1), (0, 2), (1, 2)]
    M = boundary_matrix(tri, edges)
    # (0,1) coeff +1, (0,2) coeff -1, (1,2) coeff +1
    check("triangle boundary signs", M == [[1], [-1], [1]], f"{M}")

    # ---- 11. b_0 = number of connected components ---------------------------------------
    # two disjoint edges -> 2 components
    cx = {0: [(0,), (1,), (2,), (3,)], 1: [(0, 1), (2, 3)]}
    b, t = homology(cx)
    check("two components -> b_0 = 2", b[0] == 2, f"{b}")

    # ---- 12. a single point -----------------------------------------------------------
    b, t = homology({0: [(0,)]})
    check("point Betti (1)", _betti_list(b) == [1], f"{_betti_list(b)}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
