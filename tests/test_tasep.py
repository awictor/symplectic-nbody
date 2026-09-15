"""Validate TASEP: three-phase density/current formulas, phase classifier, current bound, relaxation."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import tasep


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("TASEP tests")

    # --- phase classifier matches the (alpha, beta) regions ---
    check("low-density phase", tasep.phase(0.2, 0.8) == "low-density")
    check("high-density phase", tasep.phase(0.8, 0.2) == "high-density")
    check("maximal-current phase", tasep.phase(0.8, 0.8) == "maximal-current")
    check("LD on the coexistence-ish (alpha<beta)", tasep.phase(0.3, 0.4) == "low-density")

    # --- exact density formulas ---
    check("LD density = alpha", abs(tasep.predicted_density(0.2, 0.8) - 0.2) < 1e-12)
    check("HD density = 1-beta", abs(tasep.predicted_density(0.8, 0.2) - 0.8) < 1e-12)
    check("MC density = 1/2", abs(tasep.predicted_density(0.8, 0.8) - 0.5) < 1e-12)

    # --- exact current formulas ---
    check("LD current alpha(1-alpha)", abs(tasep.predicted_current(0.2, 0.8) - 0.2 * 0.8) < 1e-12)
    check("HD current beta(1-beta)", abs(tasep.predicted_current(0.8, 0.2) - 0.2 * 0.8) < 1e-12)
    check("MC current 1/4", abs(tasep.predicted_current(0.8, 0.8) - 0.25) < 1e-12)
    check("current <= 1/4 always", all(tasep.predicted_current(a / 10, b / 10) <= 0.25 + 1e-12
                                       for a in range(1, 10) for b in range(1, 10)))

    # --- simulated bulk density matches the exact prediction in each phase ---
    L = 100
    steps = 3_000_000
    for alpha, beta, name in [(0.25, 0.8, "LD"), (0.8, 0.25, "HD"), (0.8, 0.8, "MC")]:
        res = tasep.simulate(L, alpha, beta, steps, seed=1)
        pred = tasep.predicted_density(alpha, beta)
        check(f"{name} simulated density ~ prediction ({res['density']:.3f} vs {pred:.3f})",
              abs(res["density"] - pred) < 0.05)

    # --- simulated current matches the prediction in the low-density phase ---
    res = tasep.simulate(L, 0.25, 0.8, steps, seed=1)
    check(f"LD current ~ alpha(1-alpha) ({res['current']:.3f} vs {0.25*0.75:.3f})",
          abs(res["current"] - 0.25 * 0.75) < 0.03)

    # --- maximal-current: current near 1/4 ---
    res = tasep.simulate(L, 0.8, 0.8, steps, seed=1)
    check(f"MC current ~ 1/4 ({res['current']:.3f})", abs(res["current"] - 0.25) < 0.04)

    # --- relaxation: full and empty starts reach the same steady density ---
    d_full = tasep.simulate(80, 0.3, 0.7, 2_000_000, seed=1, init_density=0.95)["density"]
    d_empty = tasep.simulate(80, 0.3, 0.7, 2_000_000, seed=1, init_density=0.05)["density"]
    check(f"same steady state from full/empty ({d_full:.3f} vs {d_empty:.3f})",
          abs(d_full - d_empty) < 0.05)

    # --- density profile is roughly flat in the bulk (LD phase) ---
    prof = tasep.steady_state_profile(60, 0.25, 0.8, 2_000_000, seed=1)
    bulk = prof[20:40]
    m = sum(bulk) / len(bulk)
    flat = all(abs(x - m) < 0.1 for x in bulk)
    check(f"LD bulk profile flat near alpha ({m:.3f})", flat and abs(m - 0.25) < 0.06)

    # --- current continuous across coexistence line alpha = beta < 1/2 ---
    # LD current at alpha=beta=0.3 should equal HD current there (both 0.3*0.7)
    check("current continuous on coexistence", abs(tasep.predicted_current(0.3, 0.3) - 0.3 * 0.7) < 1e-12)

    # --- deterministic ---
    a = tasep.simulate(50, 0.3, 0.7, 500000, seed=42)
    b = tasep.simulate(50, 0.3, 0.7, 500000, seed=42)
    check("deterministic", a["density"] == b["density"] and a["current"] == b["current"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
