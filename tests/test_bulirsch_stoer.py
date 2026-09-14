"""Tests for Bulirsch-Stoer: exact ODE solutions, high accuracy, energy conservation, RK45 agreement."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bulirsch_stoer import (  # noqa: E402
    modified_midpoint, bulirsch_stoer_step, solve, solve_fixed,
    harmonic_oscillator, kepler_2d, harmonic_energy, kepler_energy,
)
import rk45  # noqa: E402


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
    # ---- 1. exponential y' = y, y(0)=1 -> e at t=1 --------------------------------------
    val = solve(lambda t, y: [y[0]], 0, [1.0], 1)[0]
    check("y'=y -> e at t=1", abs(val - math.e) < 1e-12, f"{val}")

    # ---- 2. y' = -y -> e^{-t} ------------------------------------------------------------
    val = solve(lambda t, y: [-y[0]], 0, [1.0], 2)[0]
    check("y'=-y -> e^-2", abs(val - math.exp(-2)) < 1e-12, f"{val}")

    # ---- 3. harmonic oscillator: x(t) = cos(t) -----------------------------------------
    f = harmonic_oscillator(1.0)
    ts, ys = solve_fixed(f, 0, [1.0, 0.0], 2 * math.pi, steps=40)
    # at t=2pi, back to (1, 0)
    check("harmonic returns to start", abs(ys[-1][0] - 1.0) < 1e-9 and abs(ys[-1][1]) < 1e-9,
          f"{ys[-1]}")
    # midpoint value: x(pi) = -1
    ts2, ys2 = solve_fixed(f, 0, [1.0, 0.0], math.pi, steps=20)
    check("harmonic x(pi) = -1", abs(ys2[-1][0] - (-1.0)) < 1e-9, f"{ys2[-1][0]}")

    # ---- 4. cos(t) matched at several points -------------------------------------------
    ok = True
    for T in [0.5, 1.0, 2.0, 3.0]:
        _, ys = solve_fixed(f, 0, [1.0, 0.0], T, steps=max(4, int(T * 8)))
        if abs(ys[-1][0] - math.cos(T)) > 1e-8:
            ok = False
            check("cos(t) matched", False, f"T={T}: {ys[-1][0]} vs {math.cos(T)}")
            break
    if ok:
        check("harmonic matches cos(t) at several T", True)

    # ---- 5. harmonic energy conserved to near machine precision -------------------------
    ts, ys = solve_fixed(f, 0, [1.0, 0.0], 20 * math.pi, steps=200)
    e0 = harmonic_energy([1.0, 0.0])
    max_drift = max(abs(harmonic_energy(y) - e0) for y in ys)
    check("harmonic energy conserved (20 periods)", max_drift < 1e-10, f"drift {max_drift:.2e}")

    # ---- 6. Kepler circular orbit returns to start -------------------------------------
    kf = kepler_2d(1.0)
    ts, ys = solve_fixed(kf, 0, [1.0, 0.0, 0.0, 1.0], 2 * math.pi, steps=100)
    check("Kepler orbit closes", abs(ys[-1][0] - 1.0) < 1e-6 and abs(ys[-1][1]) < 1e-6,
          f"{ys[-1][:2]}")

    # ---- 7. Kepler energy conserved -----------------------------------------------------
    e0 = kepler_energy([1.0, 0.0, 0.0, 1.0])
    max_drift = max(abs(kepler_energy(y) - e0) for y in ys)
    check("Kepler energy conserved", max_drift < 1e-9, f"drift {max_drift:.2e}")

    # ---- 8. extrapolation raises accuracy with more midpoint levels ---------------------
    f_exp = lambda t, y: [y[0]]
    errs = []
    for k in [2, 4, 6, 8]:
        est, _ = bulirsch_stoer_step(f_exp, 0, [1.0], 1.0, k_max=k)
        errs.append(abs(est[0] - math.e))
    check("more extrapolation levels -> smaller error",
          all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), f"{[f'{e:.1e}' for e in errs]}")

    # ---- 9. single midpoint sweep is only 2nd order (contrast) -------------------------
    mm2 = modified_midpoint(f_exp, 0, [1.0], 1.0, 2)
    mm4 = modified_midpoint(f_exp, 0, [1.0], 1.0, 4)
    err2 = abs(mm2[0] - math.e)
    err4 = abs(mm4[0] - math.e)
    # error ratio ~ 4 for a second-order method when halving substep size
    check("modified midpoint is 2nd order", 3.0 < err2 / err4 < 5.0, f"ratio {err2/err4:.2f}")

    # ---- 10. agrees with RK45 where both accurate --------------------------------------
    # solve y' = -2 t y (Gaussian), y(0)=1 -> y(t) = e^{-t^2}
    g = lambda t, y: [-2 * t * y[0]]
    ts, ys = solve_fixed(g, 0, [1.0], 1.5, steps=30)
    bs_val = ys[-1][0]
    rk_ts, rk_ys = rk45.solve(g, 0, [1.0], 1.5, atol=1e-11, rtol=1e-11)
    rk_val = rk_ys[-1][0]
    check("Bulirsch-Stoer == RK45 on Gaussian ODE", abs(bs_val - rk_val) < 1e-8,
          f"BS {bs_val} RK {rk_val}")
    check("both match e^{-t^2} at 1.5", abs(bs_val - math.exp(-1.5 ** 2)) < 1e-8)

    # ---- 11. system of two decoupled exponentials --------------------------------------
    sysf = lambda t, y: [y[0], -y[1]]
    res = solve(sysf, 0, [1.0, 1.0], 1.0)
    check("system: [e, 1/e]", abs(res[0] - math.e) < 1e-10 and abs(res[1] - math.exp(-1)) < 1e-10,
          f"{res}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
