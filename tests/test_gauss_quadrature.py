"""Tests for gauss_quadrature: exactness to degree 2n-1, moments, positivity, known integrals."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gauss_quadrature import (gauss_hermite, gauss_laguerre, integrate_hermite,  # noqa: E402
                              integrate_laguerre, gaussian_expectation)


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


def hermite_moment(k):
    """Integral x^k e^{-x^2} over (-inf, inf): 0 for odd k, (k-1)!! sqrt(pi) / 2^(k/2) for even k."""
    if k % 2 == 1:
        return 0.0
    # double factorial (k-1)!!
    df = 1.0
    j = k - 1
    while j > 0:
        df *= j
        j -= 2
    return df * math.sqrt(math.pi) / (2 ** (k // 2))


def laguerre_moment(k):
    """Integral x^k e^{-x} over [0, inf) = k! (Gamma(k+1))."""
    return float(math.factorial(k))


def main():
    # ---- 1. Gauss-Hermite exactness to degree 2n-1 ------------------------------------
    for n in (2, 3, 5, 8):
        worst = 0.0
        for k in range(2 * n):        # degrees 0 .. 2n-1
            est = integrate_hermite(lambda x, k=k: x ** k, n)
            exact = hermite_moment(k)
            worst = max(worst, abs(est - exact))
        check(f"Gauss-Hermite exact to degree {2*n-1} (n={n})", worst < 1e-8, f"worst err {worst:.2e}")

    # ---- 2. Gauss-Laguerre exactness to degree 2n-1 -----------------------------------
    for n in (2, 3, 5, 8):
        worst = 0.0
        for k in range(2 * n):
            est = integrate_laguerre(lambda x, k=k: x ** k, n)
            exact = laguerre_moment(k)
            worst = max(worst, abs(est - exact) / max(1, exact))
        check(f"Gauss-Laguerre exact to degree {2*n-1} (n={n})", worst < 1e-8, f"worst rel err {worst:.2e}")

    # ---- 3. weights positive and sum to the total mass --------------------------------
    for n in (3, 6, 10):
        _, wh = gauss_hermite(n)
        check(f"Hermite weights positive (n={n})", all(w > 0 for w in wh))
        check(f"Hermite weights sum to sqrt(pi) (n={n})", abs(sum(wh) - math.sqrt(math.pi)) < 1e-9,
              f"{sum(wh):.10f}")
        _, wl = gauss_laguerre(n)
        check(f"Laguerre weights positive (n={n})", all(w > 0 for w in wl))
        check(f"Laguerre weights sum to 1 (n={n})", abs(sum(wl) - 1.0) < 1e-9, f"{sum(wl):.10f}")

    # ---- 4. Hermite nodes are symmetric about zero ------------------------------------
    nodes, _ = gauss_hermite(7)
    nodes_sorted = sorted(nodes)
    sym = all(abs(nodes_sorted[i] + nodes_sorted[-1 - i]) < 1e-9 for i in range(len(nodes) // 2))
    check("Hermite nodes symmetric about zero", sym)
    check("odd-n Hermite has a node at zero", any(abs(x) < 1e-9 for x in nodes))

    # ---- 5. Laguerre nodes are positive -----------------------------------------------
    nodes, _ = gauss_laguerre(8)
    check("Laguerre nodes all positive", all(x > 0 for x in nodes))

    # ---- 6. known hard integrals ------------------------------------------------------
    # integral e^{-x^2} = sqrt(pi)
    check("integral e^{-x^2} = sqrt(pi)", abs(integrate_hermite(lambda x: 1.0, 3) - math.sqrt(math.pi)) < 1e-9)
    # Gamma(s+1) = integral x^s e^{-x}; use s=2.5 -> Gamma(3.5)
    s = 2.5
    got = integrate_laguerre(lambda x: x ** s, 30)
    exact = math.gamma(s + 1)
    check("Gamma(3.5) via Gauss-Laguerre", abs(got - exact) / exact < 1e-4, f"{got:.6f} vs {exact:.6f}")

    # ---- 7. Gaussian expectations -----------------------------------------------------
    # E[x^2] for N(0,1) = 1; for N(mu, sigma^2) = mu^2 + sigma^2
    check("E[x^2] N(0,1) = 1", abs(gaussian_expectation(lambda x: x * x, 0, 1, 5) - 1.0) < 1e-9)
    check("E[x^2] N(3,2) = 13", abs(gaussian_expectation(lambda x: x * x, 3, 2, 5) - (9 + 4)) < 1e-8)
    # E[e^x] for N(0,1) = e^{1/2}
    check("E[e^x] N(0,1) = sqrt(e)",
          abs(gaussian_expectation(lambda x: math.exp(x), 0, 1, 20) - math.sqrt(math.e)) < 1e-8)
    # E[1] = 1 (normalisation)
    check("E[1] = 1 (density normalised)", abs(gaussian_expectation(lambda x: 1.0, 5, 3, 4) - 1.0) < 1e-12)

    # ---- 8. convergence on a smooth non-polynomial integrand --------------------------
    # integral cos(x) e^{-x^2} = sqrt(pi) e^{-1/4}
    exact = math.sqrt(math.pi) * math.exp(-0.25)
    errs = [abs(integrate_hermite(math.cos, n) - exact) for n in (3, 5, 8, 12)]
    check("Gauss-Hermite converges on cos(x)", errs[-1] < 1e-10 and errs[-1] < errs[0],
          f"errors {[f'{e:.1e}' for e in errs]}")

    # ---- 9. validation errors ---------------------------------------------------------
    try:
        gauss_hermite(0)
        check("n<1 rejected", False)
    except ValueError:
        check("n<1 rejected", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
