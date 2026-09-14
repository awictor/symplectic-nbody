"""Validate Aberth-Ehrlich: known factors, Vieta, cross-check vs Durand-Kerner, cubic convergence."""

import cmath
import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import aberth
import durand_kerner as dk


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def match(found, expected, tol=1e-6):
    """Every expected root has a distinct nearby found root."""
    found = list(found)
    for e in expected:
        best = min(range(len(found)), key=lambda i: abs(found[i] - e))
        if abs(found[best] - e) > tol:
            return False
        found.pop(best)
    return True


def main():
    print("Aberth-Ehrlich tests")

    # --- distinct real roots ---
    expected = [1.0, -2.0, 3.0, -4.0, 5.0]
    coeffs = aberth.from_roots(expected)
    r = aberth.roots(coeffs)
    check("distinct real roots recovered", match(r, [complex(x) for x in expected], 1e-9))
    check("  residuals ~ 0", all(aberth.residual(coeffs, z) < 1e-7 for z in r))

    # --- complex-conjugate roots ---
    expected = [1 + 2j, 1 - 2j, -3 + 0j, 0 + 1j, 0 - 1j]
    coeffs = aberth.from_roots(expected)
    r = aberth.roots(coeffs)
    check("complex-conjugate roots recovered", match(r, expected, 1e-8))

    # --- clustered roots (near-degenerate, the hard case) ---
    expected = [1.0, 1.001, 0.999, 2.0]
    coeffs = aberth.from_roots(expected)
    r = aberth.roots(coeffs, max_iter=300)
    check("clustered roots recovered", match(r, [complex(x) for x in expected], 1e-4))

    # --- repeated root (double) ---
    expected = [2.0, 2.0, -1.0]
    coeffs = aberth.from_roots(expected)
    r = aberth.roots(coeffs, max_iter=400)
    check("repeated root recovered", match(r, [complex(x) for x in expected], 1e-3))

    # --- Vieta: sum and product of roots ---
    expected = [2.0, -3.0, 0.5, 4.0, -1.5]
    coeffs = aberth.from_roots(expected)   # monic, so a_n = 1
    r = aberth.roots(coeffs)
    n = len(expected)
    sum_roots = sum(r)
    prod_roots = 1 + 0j
    for z in r:
        prod_roots *= z
    # monic poly x^n + c1 x^(n-1)+...: sum = -c1, product = (-1)^n c_n
    c = aberth._normalise(coeffs)
    check("Vieta: sum of roots == -a_{n-1}/a_n", abs(sum_roots - (-c[1] / c[0])) < 1e-8)
    check("Vieta: product == (-1)^n a_0/a_n",
          abs(prod_roots - ((-1) ** n) * c[-1] / c[0]) < 1e-8)

    # --- cross-check against Durand-Kerner on the same polynomial ---
    coeffs = aberth.from_roots([1 + 1j, 1 - 1j, 2.0, -2.0, 0.3, 5.0])
    ra = sorted(aberth.roots(coeffs), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    rd = sorted(dk.roots(coeffs), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    ok = all(abs(ra[i] - rd[i]) < 1e-6 for i in range(len(ra)))
    check("matches Durand-Kerner", ok and len(ra) == len(rd))

    # --- Aberth converges in FEWER iterations than Durand-Kerner (cubic vs quadratic) ---
    coeffs = aberth.from_roots([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    _, it_ab = aberth.roots(coeffs, tol=1e-12, track=True)
    # Durand-Kerner iteration count via its own loop: replicate the convergence test
    # by comparing at matched tolerance -- Aberth should need markedly fewer sweeps.
    it_dk = _dk_iterations(coeffs, tol=1e-12)
    check(f"Aberth cubic beats DK quadratic ({it_ab} vs {it_dk} iters)", it_ab < it_dk)

    # --- reproducible per seed; different seeds still converge to the same set ---
    coeffs = aberth.from_roots([1.0, -2.0, 3 + 1j, 3 - 1j])
    r1 = aberth.roots(coeffs, seed=1)
    r2 = aberth.roots(coeffs, seed=1)
    r3 = aberth.roots(coeffs, seed=999)
    check("reproducible per seed", all(r1[i] == r2[i] for i in range(len(r1))))
    s1 = sorted(r1, key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    s3 = sorted(r3, key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    check("different seed -> same root set", all(abs(s1[i] - s3[i]) < 1e-6 for i in range(len(s1))))

    # --- degree-1 and degenerate cases ---
    check("linear root", abs(aberth.roots([2.0, -6.0])[0] - 3.0) < 1e-12)
    check("constant poly has no roots", aberth.roots([5.0]) == [])

    # --- real_roots filter ---
    rr = aberth.real_roots(aberth.from_roots([1.0, 2.0, 1 + 1j, 1 - 1j]))
    check("real_roots returns only reals", match(rr, [1.0, 2.0], 1e-6) and len(rr) == 2)

    # --- high degree (Wilkinson-like moderate) ---
    expected = [float(k) for k in range(1, 11)]  # roots 1..10
    coeffs = aberth.from_roots(expected)
    r = aberth.roots(coeffs, max_iter=300, tol=1e-13)
    check("degree-10 integer roots recovered", match(r, [complex(x) for x in expected], 1e-4))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


def _dk_iterations(coeffs, tol):
    """Count Durand-Kerner sweeps to convergence (mirrors its fixed-point update) for a fair
    iteration-count comparison against Aberth. Uses the SAME initial circle as Aberth so the
    comparison isolates the update rule (cubic vs quadratic), not the seeding."""
    mono = aberth._normalise(coeffs)
    n = len(mono) - 1
    z = aberth._initial_guesses(mono, 12345)

    def peval(x):
        p = mono[0]
        for c in mono[1:]:
            p = p * x + c
        return p

    for it in range(500):
        max_step = 0.0
        new = list(z)
        for i in range(n):
            denom = mono[0]
            for j in range(n):
                if j != i:
                    denom *= (z[i] - z[j])
            if denom == 0:
                denom = 1e-30
            w = peval(z[i]) / denom
            new[i] = z[i] - w
            max_step = max(max_step, abs(w))
        z = new
        if max_step < tol:
            return it + 1
    return 500


if __name__ == "__main__":
    main()
