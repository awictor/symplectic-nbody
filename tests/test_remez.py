"""Tests for Remez minimax: degree-0 midrange, equioscillation, beats Chebyshev/least-squares."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remez import (  # noqa: E402
    remez,
    poly_eval,
    max_error,
    equioscillation_points,
    chebyshev_interp,
    least_squares_poly,
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
    # ---- 1. degree-0 minimax constant is the midrange (max+min)/2 -----------------------
    # for a monotone f on [a,b], min=f(a), max=f(b); best constant = (f(a)+f(b))/2,
    # error = (f(b)-f(a))/2.
    f = math.exp
    a, b = 0.0, 1.0
    coeffs, level, refs = remez(f, a, b, 0)
    mid = (math.exp(0.0) + math.exp(1.0)) / 2
    half = (math.exp(1.0) - math.exp(0.0)) / 2
    check("degree-0 constant = midrange", abs(coeffs[0] - mid) < 1e-9, f"{coeffs[0]} vs {mid}")
    check("degree-0 level = half-range", abs(level - half) < 1e-9, f"{level} vs {half}")

    # ---- 2. equioscillation: at least degree+2 alternating extrema at level E -----------
    for degree in [1, 2, 3, 4, 5]:
        coeffs, level, refs = remez(math.exp, 0.0, 2.0, degree)
        pts = equioscillation_points(math.exp, coeffs, 0.0, 2.0)
        # keep only the near-maximal extrema
        big = [(x, e) for x, e in pts if abs(e) > 0.9 * level]
        # signs must alternate
        alt = all(_opp(big[i][1], big[i + 1][1]) for i in range(len(big) - 1))
        check(f"deg {degree}: >= n+2 equioscillation points",
              len(big) >= degree + 2 and alt, f"{len(big)} extrema, alt={alt}")

    # ---- 3. all near-maximal extrema have magnitude ~ E ---------------------------------
    coeffs, level, refs = remez(math.sin, 0.0, math.pi, 4)
    pts = equioscillation_points(math.sin, coeffs, 0.0, math.pi)
    big = [abs(e) for _, e in pts if abs(e) > 0.9 * level]
    check("extrema all reach level E", all(abs(e - level) < 1e-6 for e in big),
          f"levels {[round(e, 9) for e in big]}")

    # ---- 4. minimax beats Chebyshev interpolation on sup norm ---------------------------
    for fn, a, b, degree in [(math.exp, 0.0, 2.0, 4),
                             (math.sin, 0.0, math.pi, 5),
                             (lambda x: 1.0 / (1 + x * x), -2.0, 4.0, 6)]:
        mcoef, mlevel, _ = remez(fn, a, b, degree)
        merr = max_error(fn, mcoef, a, b)
        ccoef = chebyshev_interp(fn, a, b, degree)
        cerr = max_error(fn, ccoef, a, b)
        check("minimax <= Chebyshev interp sup error", merr <= cerr + 1e-9,
              f"minimax {merr:.3e} vs cheby {cerr:.3e}")

    # ---- 5. minimax beats least-squares on sup norm -------------------------------------
    for fn, a, b, degree in [(math.exp, 0.0, 2.0, 3),
                             (lambda x: abs(x), -1.0, 1.0, 4)]:
        mcoef, mlevel, _ = remez(fn, a, b, degree, grid=4001)
        merr = max_error(fn, mcoef, a, b)
        lcoef = least_squares_poly(fn, a, b, degree)
        lerr = max_error(fn, lcoef, a, b)
        check("minimax <= least-squares sup error", merr <= lerr + 1e-6,
              f"minimax {merr:.3e} vs ls {lerr:.3e}")

    # ---- 6. brute-force: no small coefficient perturbation lowers the sup error ---------
    fn = math.exp
    a, b, degree = 0.0, 1.0, 2
    mcoef, mlevel, _ = remez(fn, a, b, degree)
    base = max_error(fn, mcoef, a, b, grid=2001)
    improved = False
    step = base * 0.02
    # search a small grid around each coefficient
    for i in range(len(mcoef)):
        for d in (-step, -step / 2, step / 2, step):
            trial = list(mcoef)
            trial[i] += d
            if max_error(fn, trial, a, b, grid=2001) < base - 1e-12:
                improved = True
    check("no local perturbation beats minimax", not improved)

    # ---- 7. level E matches the measured sup error --------------------------------------
    mcoef, mlevel, _ = remez(math.cos, 0.0, 3.0, 5)
    err = max_error(math.cos, mcoef, 0.0, 3.0)
    check("returned level == sup error", abs(mlevel - err) < 1e-6, f"{mlevel} vs {err}")

    # ---- 8. exact polynomial recovered exactly ------------------------------------------
    poly = [1.0, -2.0, 0.5]  # 1 - 2x + 0.5 x^2

    def pf(x):
        return poly_eval(poly, x)
    mcoef, mlevel, _ = remez(pf, -1.0, 1.0, 2)
    check("recovers exact polynomial (error ~ 0)", mlevel < 1e-9, f"level {mlevel}")
    check("recovered coeffs match", all(abs(mcoef[i] - poly[i]) < 1e-7 for i in range(3)),
          f"{mcoef}")

    # ---- 9. higher degree -> smaller minimax error --------------------------------------
    errs = []
    for degree in [2, 4, 6, 8]:
        c, lvl, _ = remez(math.exp, 0.0, 2.0, degree)
        errs.append(lvl)
    check("minimax error decreases with degree",
          all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), f"{errs}")

    # ---- 10. input validation -----------------------------------------------------------
    try:
        remez(math.exp, 1.0, 0.0, 2)
        check("a >= b raises", False)
    except ValueError:
        check("a >= b raises", True)
    try:
        remez(math.exp, 0.0, 1.0, -1)
        check("negative degree raises", False)
    except ValueError:
        check("negative degree raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


def _opp(u, v):
    return (u >= 0) != (v >= 0)


if __name__ == "__main__":
    main()
