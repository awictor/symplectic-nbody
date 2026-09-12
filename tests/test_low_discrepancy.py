"""Tests for low_discrepancy: hand-checked radical inverse, discrepancy beats random, QMC beats MC."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from low_discrepancy import (radical_inverse, van_der_corput, first_primes,  # noqa: E402
                             halton, hammersley, star_discrepancy, qmc_integrate)


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


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def main():
    # ---- 1. hand-computed radical inverse ---------------------------------------------
    check("phi_2(1) = 0.5", abs(radical_inverse(1, 2) - 0.5) < 1e-12)
    check("phi_2(2) = 0.25", abs(radical_inverse(2, 2) - 0.25) < 1e-12)
    check("phi_2(3) = 0.75", abs(radical_inverse(3, 2) - 0.75) < 1e-12)
    check("phi_2(4) = 0.125", abs(radical_inverse(4, 2) - 0.125) < 1e-12)
    check("phi_3(1) = 1/3", abs(radical_inverse(1, 3) - 1.0 / 3) < 1e-12)
    check("phi_3(3) = 1/9", abs(radical_inverse(3, 3) - 1.0 / 9) < 1e-12)
    check("phi_b(0) = 0", radical_inverse(0, 5) == 0.0)

    try:
        radical_inverse(5, 1)
        check("base < 2 raises", False)
    except ValueError:
        check("base < 2 raises", True)

    # ---- 2. van der Corput stratification: first b^m points permute the grid ----------
    for base, m in ((2, 4), (3, 3), (5, 2)):
        bm = base ** m
        vals = van_der_corput(bm, base, start=0)  # include index 0 -> gives {0, 1/bm, ...}
        grid = sorted(round(v * bm) for v in vals)
        check(f"van der Corput base {base} first {bm} points stratify",
              grid == list(range(bm)), f"{grid[:6]}...")

    # ---- 3. all points inside the unit cube -------------------------------------------
    pts = halton(500, 3)
    inside = all(all(0 <= c < 1 for c in p) for p in pts)
    check("Halton points inside [0,1)^3", inside)
    check("first primes correct", first_primes(5) == [2, 3, 5, 7, 11])

    # ---- 4. Halton discrepancy beats pseudo-random ------------------------------------
    N = 256
    h = halton(N, 2)
    rng = LCG(2024)
    rand = [(rng.u(), rng.u()) for _ in range(N)]
    dh = star_discrepancy(h, samples=1500)
    dr = star_discrepancy(rand, samples=1500)
    check("Halton discrepancy < pseudo-random discrepancy", dh < dr, f"halton={dh:.4f} rand={dr:.4f}")

    # discrepancy shrinks as N grows
    d_small = star_discrepancy(halton(64, 2), samples=1000)
    d_large = star_discrepancy(halton(512, 2), samples=1000)
    check("Halton discrepancy shrinks with N", d_large < d_small, f"{d_large:.4f} < {d_small:.4f}")

    # ---- 5. QMC integration of known integrals ----------------------------------------
    # integral of x over [0,1] = 0.5
    est = qmc_integrate(lambda p: p[0], 1, 2000)
    check("QMC integral of x = 0.5", abs(est - 0.5) < 1e-3, f"{est}")

    # integral of x*y over [0,1]^2 = 0.25
    est2 = qmc_integrate(lambda p: p[0] * p[1], 2, 4000)
    check("QMC integral of x*y = 0.25", abs(est2 - 0.25) < 2e-3, f"{est2}")

    # quarter unit disk area = pi/4 (indicator that x^2+y^2 < 1)
    est_pi = qmc_integrate(lambda p: 1.0 if p[0] ** 2 + p[1] ** 2 < 1 else 0.0, 2, 8000) * 4
    check("QMC estimates pi via quarter disk", abs(est_pi - math.pi) < 0.02, f"pi~{est_pi:.4f}")

    # ---- 6. QMC beats plain MC at the same N (on average over the smooth integrand) ---
    # true integral of exp(x*y) over [0,1]^2
    def f(p):
        return math.exp(p[0] * p[1])
    # reference by dense grid
    G = 400
    ref = sum(math.exp((i + 0.5) / G * (j + 0.5) / G) for i in range(G) for j in range(G)) / (G * G)

    Nq = 1000
    qmc_est = qmc_integrate(f, 2, Nq)
    qmc_err = abs(qmc_est - ref)

    # average MC error over several seeds
    mc_errs = []
    for seed in range(12):
        r = LCG(seed * 101 + 1)
        s = sum(f((r.u(), r.u())) for _ in range(Nq)) / Nq
        mc_errs.append(abs(s - ref))
    mc_err = sum(mc_errs) / len(mc_errs)
    check("QMC error < average MC error at same N", qmc_err < mc_err,
          f"qmc={qmc_err:.5f} mc_avg={mc_err:.5f}")

    # ---- 7. Hammersley basics ----------------------------------------------------------
    ham = hammersley(100, 2)
    check("Hammersley has right count and dim", len(ham) == 100 and len(ham[0]) == 2)
    check("Hammersley first coord is i/N", abs(ham[10][0] - 10 / 100) < 1e-12)
    check("Hammersley points inside cube", all(0 <= c < 1 for p in ham for c in p))
    # Hammersley discrepancy also beats random
    dham = star_discrepancy(ham, samples=1200)
    check("Hammersley discrepancy < random", dham < dr, f"ham={dham:.4f} rand={dr:.4f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
