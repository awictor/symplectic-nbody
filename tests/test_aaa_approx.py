"""Validate AAA: interpolation, machine-precision smooth fits, pole recovery, beating polynomials."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import aaa_approx as aaa


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def max_err(r, f, xs):
    return max(abs(r["eval"](x) - f(x)) for x in xs)


def poly_lsq_fit(xs, ys, degree):
    """Least-squares polynomial fit; returns a callable. Solves the normal equations."""
    n = degree + 1
    # Vandermonde normal equations
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum(ys[k] * xs[k] ** i for k in range(len(xs))) for i in range(n)]
    c = _dense_solve(A, b)

    def p(x):
        return sum(c[i] * x ** i for i in range(n))
    return p


def _dense_solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / pv
            for cc in range(col, n + 1):
                M[r][cc] -= f * M[col][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


def main():
    print("AAA approximation tests")

    # --- interpolates the support data exactly ---
    Z = linspace(-1, 1, 40)
    f = math.exp
    F = [f(z) for z in Z]
    r = aaa.aaa(Z, F, tol=1e-13)
    exact = all(abs(r["eval"](r["support_z"][j]) - r["support_f"][j]) < 1e-12
                for j in range(len(r["support_z"])))
    check("interpolates support points exactly", exact)

    # --- smooth exp to near machine precision with few terms ---
    check("exp fit near machine precision", max_err(r, f, Z) < 1e-11)
    check("  used few support points", r["num_terms"] <= 12)

    # --- a Gaussian, likewise ---
    g = lambda x: math.exp(-x * x)
    Fg = [g(z) for z in Z]
    rg = aaa.aaa(Z, Fg, tol=1e-12)
    check("gaussian fit near machine precision", max_err(rg, g, Z) < 1e-10)

    # --- function with a pole: 1/(x - 1.5) on [-1,1], and RECOVER the pole ---
    a = 1.5
    fp = lambda x: 1.0 / (x - a)
    Zp = linspace(-1, 1, 60)
    Fp = [fp(z) for z in Zp]
    rp = aaa.aaa(Zp, Fp, tol=1e-12)
    check("pole function fit accurately", max_err(rp, fp, Zp) < 1e-9)
    pls = aaa.poles(rp)
    near = min((abs(p - a) for p in pls), default=1e9)
    check("recovers the pole at 1.5", near < 1e-6)

    # --- tan(x) on [-1.2, 1.2]: two poles just outside at +-pi/2 ---
    ft = math.tan
    Zt = linspace(-1.2, 1.2, 80)
    Ft = [ft(z) for z in Zt]
    rt = aaa.aaa(Zt, Ft, tol=1e-11)
    check("tan fit accurate", max_err(rt, ft, Zt) < 1e-7)
    plt = aaa.poles(rt)
    # nearest recovered pole to +pi/2
    d_plus = min((abs(p - math.pi / 2) for p in plt), default=1e9)
    check("recovers tan pole near pi/2", d_plus < 1e-3)

    # --- AAA beats a least-squares polynomial of equal order on a near-singular function ---
    fs = lambda x: 1.0 / (x - 1.1)
    Zs = linspace(-1, 1, 50)
    Fs = [fs(z) for z in Zs]
    rs = aaa.aaa(Zs, Fs, tol=1e-10)
    deg = rs["num_terms"]  # comparable number of parameters
    xs_test = linspace(-1, 1, 300)
    aaa_err = max(abs(rs["eval"](x) - fs(x)) for x in xs_test)
    p = poly_lsq_fit(Zs, Fs, degree=deg)
    poly_err = max(abs(p(x) - fs(x)) for x in xs_test)
    check(f"AAA beats degree-{deg} polynomial ({aaa_err:.1e} < {poly_err:.1e})", aaa_err < poly_err)

    # --- error decreases as more support points are added ---
    errs = r["errors"]
    # not strictly monotone every step, but the trend must fall a lot end-to-start
    check("error decreases with more terms", errs[-1] < errs[0] * 1e-3)

    # --- deterministic ---
    r2 = aaa.aaa(Z, F, tol=1e-13)
    check("deterministic", r2["support_z"] == r["support_z"] and r2["weights"] == r["weights"])

    # --- constant data: exact ---
    rc = aaa.aaa(linspace(0, 1, 10), [3.0] * 10, tol=1e-12)
    check("constant data reproduced", abs(rc["eval"](0.37) - 3.0) < 1e-10)

    # --- min-singular-vector helper: residual small for a known rank-deficient matrix ---
    A = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0001], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    v = aaa._min_singular_vector(A)
    nv = math.sqrt(sum(x * x for x in v))
    check("min-singular-vector is unit length", abs(nv - 1.0) < 1e-9)
    # A v should have smaller norm than A times a generic unit vector
    Av = [sum(A[i][j] * v[j] for j in range(3)) for i in range(4)]
    nAv = math.sqrt(sum(x * x for x in Av))
    e = [1.0, 0.0, 0.0]
    Ae = [sum(A[i][j] * e[j] for j in range(3)) for i in range(4)]
    nAe = math.sqrt(sum(x * x for x in Ae))
    check("min-singular-vector minimizes ||Av||", nAv < nAe)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
