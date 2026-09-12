"""Tests for lll: reduced-basis conditions, lattice preservation, brute-force shortest, relations."""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lll import (lll_reduce, is_reduced, gram_determinant, shortest_vector,  # noqa: E402
                 integer_relation, vector_norm_sq, gram_schmidt, _brute_shortest, _round_half)


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


def det2(basis):
    """Exact integer determinant of a square integer basis, by Gram determinant, when it's a square."""
    return gram_determinant(basis)


def main():
    # ---- 1. _round_half banker's rounding -------------------------------------------
    check("round 1/2 -> 0 (even)", _round_half(Fraction(1, 2)) == 0)
    check("round 3/2 -> 2 (even)", _round_half(Fraction(3, 2)) == 2)
    check("round 5/2 -> 2 (even)", _round_half(Fraction(5, 2)) == 2)
    check("round -1/2 -> 0", _round_half(Fraction(-1, 2)) == 0)
    check("round 7/3 -> 2", _round_half(Fraction(7, 3)) == 2)
    check("round 2/3 -> 1", _round_half(Fraction(2, 3)) == 1)

    # ---- 2. classic textbook lattice: reduction produces a reduced basis --------------
    basis = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    red = lll_reduce(basis)
    check("classic basis reduces to a reduced basis", is_reduced(red))
    # lattice preserved: same Gram determinant (|det|^2)
    check("Gram determinant preserved", det2(basis) == det2(red),
          f"{det2(basis)} vs {det2(red)}")
    # reduced vectors should be shorter (sum of norms not larger)
    orig_norm = sum(vector_norm_sq(v) for v in basis)
    red_norm = sum(vector_norm_sq(v) for v in red)
    check("reduced basis is not longer overall", red_norm <= orig_norm)

    # ---- 3. a nastily-skewed 2D lattice ------------------------------------------------
    skew = [[201, 37], [98, 18]]
    red2 = lll_reduce(skew)
    check("skewed 2D basis reduces", is_reduced(red2))
    check("skewed 2D Gram det preserved", det2(skew) == det2(red2))

    # ---- 4. shortest vector matches brute force on small lattices ----------------------
    # (lattice, brute bound). 2D lattices with tiny determinant need large integer coefficients to
    # reach their true shortest vector, so their brute box must be wide; 3D boxes stay small for speed.
    lattices = [
        ([[1, 1, 1], [-1, 0, 2], [3, 5, 6]], 4),
        ([[201, 37], [98, 18]], 260),
        ([[2, 0, 0], [0, 3, 0], [1, 1, 5]], 4),
        ([[15, 23], [11, 17]], 260),
    ]
    for idx, (lat, bnd) in enumerate(lattices):
        sv = shortest_vector(lat)
        brutev, brute_n = _brute_shortest(lat, bound=bnd)
        check(f"LLL shortest matches brute force (lattice {idx})",
              vector_norm_sq(sv) == brute_n,
              f"lll={vector_norm_sq(sv)} brute={brute_n}")

    # ---- 5. lattice membership: reduced vectors are integer combos of the original -----
    # For a 2x2 unimodular-related pair, check that solving B_orig^-1 * reduced gives integers.
    def is_integer_combo_2d(orig, red):
        (a, b), (c, d) = orig
        det = a * d - b * c
        if det == 0:
            return True
        for v in red:
            x, y = v
            # solve [a c; b d]^T? we want integer p,q with p*orig0 + q*orig1 = v
            # [a b][p]   [x]      p = (d*x - c*y)/det ... using rows as basis vectors
            # [c d][q] = [y]
            p_num = d * x - c * y
            q_num = -b * x + a * y
            if p_num % det != 0 or q_num % det != 0:
                return False
        return True

    check("reduced 2D vectors are integer combinations of the original",
          is_integer_combo_2d(skew, red2))

    # ---- 6. delta validation -----------------------------------------------------------
    try:
        lll_reduce(basis, delta=Fraction(1, 5))
        check("delta out of range raises", False)
    except ValueError:
        check("delta out of range raises", True)

    # higher delta still gives a reduced (and valid) basis
    redhi = lll_reduce(basis, delta=Fraction(99, 100))
    check("high-delta basis reduced", is_reduced(redhi, delta=Fraction(99, 100)))
    check("high-delta Gram det preserved", det2(basis) == det2(redhi))

    # ---- 7. integer relation detection -------------------------------------------------
    # 2*x0 - 3*x1 = 0 exactly when x1 = (2/3) x0
    x0, x1 = 3.0, 2.0     # 2*3 - 3*2 = 0
    rel = integer_relation([x0, x1], scale=10 ** 8)
    # relation should be proportional to (2, -3) or (-2, 3): a0*3 + a1*2 == 0
    check("integer relation recovers 2*3 - 3*2 = 0 form",
          rel is not None and rel[0] * x0 + rel[1] * x1 == 0 and (rel[0], rel[1]) != (0, 0),
          f"rel={rel}")

    # a relation among three: 1*6 + 2*3 - 3*4 = 0  -> reals 6,3,4 with coeffs (1,2,-3)
    reals = [6.0, 3.0, 4.0]
    rel3 = integer_relation(reals, scale=10 ** 8)
    check("integer relation finds a true relation among three reals",
          rel3 is not None and abs(sum(a * r for a, r in zip(rel3, reals))) < 1e-6
          and any(a != 0 for a in rel3),
          f"rel3={rel3} residual={sum(a*r for a,r in zip(rel3, reals)) if rel3 else None}")

    # no small relation among 1 and an irrational-ish value: coefficients shouldn't be tiny-and-exact
    # (we just check the routine returns something without crashing)
    rel_none = integer_relation([1.0, 1.4142135623730951], scale=10 ** 6)
    check("integer relation runs on incommensurate inputs", rel_none is not None)

    # ---- 8. identity lattice is already reduced ----------------------------------------
    ident = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    check("identity lattice already reduced", is_reduced(ident))
    check("identity reduces to itself (up to order/sign)",
          {tuple(sorted(map(abs, v))) for v in lll_reduce(ident)} ==
          {tuple(sorted(map(abs, v))) for v in ident})

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
