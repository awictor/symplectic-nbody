"""Validate stochastic SIR: sub/super-threshold behavior, final-size equation, extinction ~ 1/R0, conservation."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import sir_stochastic as sir


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def mean(xs):
    return sum(xs) / len(xs)


def main():
    print("Stochastic SIR tests")

    # --- R0 formula ---
    check("R0 == beta/gamma", abs(sir.r0(0.6, 0.2) - 3.0) < 1e-12)

    # --- population conserved on every path ---
    n = 500
    res = sir.simulate(n, 5, 0.5, 0.2, seed=1)
    ok = all(res["S"][t] + res["I"][t] + res["R"][t] == n for t in range(len(res["S"])))
    check("S + I + R == N conserved", ok)

    # --- below threshold (R0 < 1): outbreaks stay tiny ---
    n = 1000
    sizes = [sir.simulate(n, 5, 0.15, 0.3, seed=r + 1)["final_size"] for r in range(200)]  # R0 = 0.5
    check(f"R0<1: mean final size small ({mean(sizes):.1f})", mean(sizes) < 30)

    # --- above threshold: final size is bimodal (minor + major clusters) ---
    n = 2000
    R0 = 2.5
    beta, gamma = 0.5, 0.2  # R0 = 2.5
    sizes = [sir.simulate(n, 3, beta, gamma, seed=r + 1)["final_size"] for r in range(300)]
    minor = [s for s in sizes if s < 0.05 * n]
    major = [s for s in sizes if s >= 0.05 * n]
    check(f"R0>1: bimodal outcomes ({len(minor)} minor, {len(major)} major)",
          len(minor) > 10 and len(major) > 10)

    # --- major-outbreak fraction matches the final-size equation 1 - z = e^{-R0 z} ---
    z = sir.final_size_fraction(R0)
    major_frac = mean(major) / n
    check(f"major outbreak fraction ~ final-size eq ({major_frac:.3f} vs {z:.3f})",
          abs(major_frac - z) < 0.05)

    # --- final-size equation: z solves 1 - z = exp(-R0 z) ---
    check("final-size eq is a fixed point", abs((1 - z) - math.exp(-R0 * z)) < 1e-9)
    check("R0<=1 gives z=0", sir.final_size_fraction(0.8) == 0.0)
    check("larger R0 -> larger final size", sir.final_size_fraction(3.0) > sir.final_size_fraction(1.5))
    # R0 -> infinity, z -> 1
    check("huge R0 -> nearly everyone", sir.final_size_fraction(20.0) > 0.99)

    # --- minor-outbreak probability from one infective ~ 1/R0 ---
    check("minor prob formula 1/R0", abs(sir.minor_outbreak_probability(2.5, 1) - 1 / 2.5) < 1e-12)
    check("minor prob i0=2 is (1/R0)^2", abs(sir.minor_outbreak_probability(2.0, 2) - 0.25) < 1e-12)
    check("R0<=1 minor prob 1", sir.minor_outbreak_probability(0.7) == 1.0)

    # --- empirical extinction (minor) probability from one infective ~ 1/R0 ---
    n = 3000
    emp_ext = sir.extinction_probability(n, 1, 0.5, 0.2, n_runs=1000, seed=1)  # R0 = 2.5
    check(f"empirical minor prob ~ 1/R0 ({emp_ext:.3f} vs {1/2.5:.3f})", abs(emp_ext - 1 / 2.5) < 0.08)

    # --- larger R0 -> higher take-off (lower extinction) probability ---
    ext_lo = sir.extinction_probability(2000, 1, 0.3, 0.2, n_runs=500, seed=1)   # R0 = 1.5
    ext_hi = sir.extinction_probability(2000, 1, 0.8, 0.2, n_runs=500, seed=1)   # R0 = 4.0
    check(f"larger R0 -> lower extinction ({ext_hi:.3f} < {ext_lo:.3f})", ext_hi < ext_lo)

    # --- deterministic final size ---
    check("deterministic final size = N*z", abs(sir.deterministic_final_size(1000, 2.5) - 1000 * z) < 1e-6)

    # --- epidemic ends (I reaches 0) ---
    res = sir.simulate(500, 5, 0.5, 0.2, seed=3)
    check("epidemic terminates (I=0)", res["I"][-1] == 0)

    # --- peak infected is at least i0 ---
    check("peak >= i0", res["peak_infected"] >= 5)

    # --- deterministic per seed ---
    a = sir.simulate(500, 5, 0.5, 0.2, seed=42)
    b = sir.simulate(500, 5, 0.5, 0.2, seed=42)
    check("reproducible per seed", a["final_size"] == b["final_size"] and a["times"] == b["times"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
